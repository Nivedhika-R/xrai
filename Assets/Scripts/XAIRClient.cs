using MagicLeap;
using SimpleJson;
using System;
using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

public class XAIRClient : Singleton<XAIRClient>
{
    [Header("Server Settings")]
    [SerializeField]
    private string ipAddress = "0.0.0.0";
    [SerializeField]
    private string port = "8000";
    [SerializeField]
    [Tooltip("Set to true if using HTTP instead of HTTPS")]
    private bool noSSL = false;

    [Header("Managers")]
    [SerializeField]
    private ServerCommunication _serverCommunication;
    [SerializeField]
    private MediaManager _mediaManager;
    [SerializeField]
    private TriangleMeshing _meshManager;

    [SerializeField]
    private TextMeshProUGUI LLMResponseText = null;
    private float _sendImagetimer = 0.0f;
    [SerializeField]
    private float _sendImageFreqHz = 1.0f;

    [SerializeField]
    private bool enableDebug = false;

    private string _serverUri;
    private bool connected = false;

    public Action<bool, string> OnWebRTCConnectionChanged;

    private readonly List<GameObject> activeTextObjects = new();

    public static Matrix4x4 StringToMatrix(string matrixStr)
    {
        string[] parts = matrixStr.Trim('[', ']').Split(',');
        if (parts.Length != 16)
        {
            Debug.LogError("Invalid matrix length.");
            return Matrix4x4.identity;
        }

        float[] values = new float[16];
        for (int i = 0; i < 16; i++)
        {
            values[i] = float.Parse(parts[i]);
        }

        Matrix4x4 matrix;
        matrix.m00 = values[0];  matrix.m01 = values[1];  matrix.m02 = values[2];  matrix.m03 = values[3];
        matrix.m10 = values[4];  matrix.m11 = values[5];  matrix.m12 = values[6];  matrix.m13 = values[7];
        matrix.m20 = values[8];  matrix.m21 = values[9];  matrix.m22 = values[10]; matrix.m23 = values[11];
        matrix.m30 = values[12]; matrix.m31 = values[13]; matrix.m32 = values[14]; matrix.m33 = values[15];

        return matrix;
    }

    void Start() {
        _serverUri = noSSL ? "http://" : "https://" + ipAddress + ":" + port;
        Debug.Log("XAIRClient starting at " + _serverUri);

        // "origin"
        if (enableDebug) {
            GameObject marker = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            marker.transform.localPosition = new Vector3(0.0f, 1.6f, 0.0f);
            marker.transform.localScale = new Vector3(0.02f, 0.02f, 0.02f);
        }

        WebRTCController.Instance.OnWebRTCConnectionStateChange += OnWebRTCConnectionStateChanged;
        WebRTCController.Instance.OnDataChannelMessageReceived += OnDataChannelMessageReceived;

        // try connect every 1 second
        StartCoroutine(Connect());
    }

    // every 1 second send image to server
    void Update()
    {
        if (_serverCommunication == null || !connected)
        {
            return;
        }

        _sendImagetimer += Time.deltaTime;
        if (_sendImagetimer >= 1.0f / _sendImageFreqHz)
        {
            _sendImagetimer = 0.0f;
            byte[] imageBytes = _mediaManager.GetImage(
                                        out Matrix4x4 cameraToWorldMatrix,
                                        out Matrix4x4 instrinsics,
                                        out _, // ignore distortion
                                        true);
            if (imageBytes != null)
            {
                // offset y coordinate by 1.6
                cameraToWorldMatrix.m13 += 1.6f;
                // send image to server
                _serverCommunication.SendImage(imageBytes, cameraToWorldMatrix, instrinsics);
            }
        }
    }

    private IEnumerator Connect()
    {
        // try connect every 1 second
        while (!MediaManager.Instance.IsMediaReady() || !connected)
        {
            yield return new WaitForSeconds(1);
            OnWebRTCConnectionChanged?.Invoke(true, _serverUri);
        }
    }

    private void OnWebRTCConnectionStateChanged(WebRTCController.WebRTCConnectionState connectionState)
    {
        Debug.Log("WebRTC connection state changed to " + connectionState);
        if (connectionState == WebRTCController.WebRTCConnectionState.Connected)
        {
            connected = true;
        }
        else if (connectionState == WebRTCController.WebRTCConnectionState.NotConnected)
        {
            connected = false;
        }
    }

    private void OnDataChannelMessageReceived(string message)
    {
        SimpleJson.SimpleJson.TryDeserializeObject(message, out object obj);
        JsonObject jsonObj = (JsonObject)obj;
        if (jsonObj.ContainsKey("type"))
        {
            string msgType = jsonObj["type"].ToString();
            if (msgType == "greeting")
            {
                string greeting = jsonObj["content"].ToString();
                UpdateLLMResponseText(greeting);
            }
            else
            if (msgType == "llm_reply")
            {
                // parse json
                string clientID = jsonObj["clientID"].ToString();
                string timestamp = jsonObj["timestamp"].ToString();

                string llmReply = jsonObj["content"].ToString();

                uint imageWidth = uint.Parse(jsonObj["imageWidth"].ToString());
                uint imageHeight = uint.Parse(jsonObj["imageHeight"].ToString());
                Vector2 imageDimensions = new(imageWidth, imageHeight);

                Matrix4x4 cameraToWorldMatrix = StringToMatrix(jsonObj["extrinsics"].ToString());
                Matrix4x4 instrinsics = StringToMatrix(jsonObj["instrinsics"].ToString());

                var objectLabelsArray = jsonObj["objectLabels"] as JsonArray;
                List<string> objectLabels = new();
                foreach (var label in objectLabelsArray)
                {
                    if (label is string labelStr)
                    {
                        objectLabels.Add(labelStr);
                    }
                }

                var objectCentersArray = jsonObj["objectCenters"] as JsonArray;
                List<Vector2> objectCenters = new();
                foreach (var center in objectCentersArray)
                {
                    if (center is JsonArray coords && coords.Count == 2)
                    {
                        float x = float.Parse(coords[0].ToString());
                        float y = float.Parse(coords[1].ToString());
                        objectCenters.Add(new Vector2(x, y));
                    }
                }

                // destroy old text objects
                for (int i = 0; i < activeTextObjects.Count; i++)
                {
                    Destroy(activeTextObjects[i]);
                }
                activeTextObjects.Clear();

                // project object centers (image space) into world space
                for (int i = 0; i < objectCenters.Count; i++)
                {
                    string objectLabel = objectLabels[i];
                    Vector2 pixelCoords = objectCenters[i];
                    if (RaycastToMesh(pixelCoords, imageDimensions, cameraToWorldMatrix, instrinsics, out Vector3 hitPoint))
                    {
                        GameObject textObject = new("LabelText");
                        textObject.transform.localScale = new Vector3(0.125f, 0.125f, 0.125f);
                        textObject.transform.position = hitPoint;

                        TextMesh textMesh = textObject.AddComponent<TextMesh>();
                        textMesh.text = objectLabel;
                        textMesh.font = Resources.GetBuiltinResource<Font>("Arial.ttf");
                        textMesh.fontSize = 250;
                        textMesh.characterSize = 0.01f;
                        textMesh.color = Color.green;
                        textMesh.alignment = TextAlignment.Center;
                        textMesh.anchor = TextAnchor.MiddleCenter;

                        textObject.transform.LookAt(Camera.main.transform);
                        textObject.transform.Rotate(0, 180, 0);

                        activeTextObjects.Add(textObject);

                        if (enableDebug) {
                            Vector3 cameraPositionWorld = cameraToWorldMatrix.GetColumn(3);
                            LineRenderer lr = new GameObject("RayLine").AddComponent<LineRenderer>();
                            lr.positionCount = 2;
                            lr.SetPosition(0, cameraPositionWorld);
                            lr.SetPosition(1, hitPoint);
                            lr.startWidth = lr.endWidth = 0.01f;
                        }
                    }
                    else if (enableDebug)
                    {
                        Debug.LogWarning("Raycast failed to hit mesh.");
                    }
                }

                UpdateLLMResponseText(llmReply);
            }
        }
    }

    private bool RaycastToMesh(Vector2 pixelCoords, Vector2 imageDimensions, Matrix4x4 cameraToWorldMatrix, Matrix4x4 instrinsics , out Vector3 hitPoint)
    {
        float fx = instrinsics.m00;
        float fy = instrinsics.m11;
        float cx = instrinsics.m02;
        float cy = instrinsics.m12;

        // image space to ndc (normalized device coordinates) (-1 to 1)
        float x = (pixelCoords.x - cx) / fx;
        float y = ((imageDimensions.y - pixelCoords.y) - cy) / fy;

        Vector3 cameraSpacePoint = new(x, y, 1.0f);

        // project to world space
        Vector3 rayDirWorld = cameraToWorldMatrix.MultiplyVector(cameraSpacePoint);

        // raycast from camera to object
        Vector3 cameraPositionWorld = cameraToWorldMatrix.GetColumn(3);
        Ray ray = new(cameraPositionWorld, rayDirWorld);
        return _meshManager.RayCastToMesh(ray, out hitPoint);
    }

    private void UpdateLLMResponseText(string text)
    {
        if (LLMResponseText != null)
        {
            LLMResponseText.text = text;
        }
    }
};
