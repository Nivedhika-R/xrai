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
    private string ipAddress = "127.0.0.1";
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
    // [SerializeField]
    // private TriangleMeshing _meshManager;

    [SerializeField]
    private AudioSource dingSource;

    [SerializeField]
    private TextMeshPro LLMResponseText = null;
    private float _sendImagetimer = 0.0f;
    [SerializeField]
    private float _sendImageFreqHz = 5f;

    [SerializeField]
    private bool enableDebug = false;

    private string _serverUri;
    private bool connected = false;

    public Action<bool, string> OnWebRTCConnectionChanged;

    private readonly List<GameObject> activeObjects = new();

    // private Material transparentMat;

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
        matrix.m00 = values[0]; matrix.m01 = values[1]; matrix.m02 = values[2]; matrix.m03 = values[3];
        matrix.m10 = values[4]; matrix.m11 = values[5]; matrix.m12 = values[6]; matrix.m13 = values[7];
        matrix.m20 = values[8]; matrix.m21 = values[9]; matrix.m22 = values[10]; matrix.m23 = values[11];
        matrix.m30 = values[12]; matrix.m31 = values[13]; matrix.m32 = values[14]; matrix.m33 = values[15];

        return matrix;
    }

    void Start()
    {
        _serverUri = noSSL ? "http://" : "https://" + ipAddress + ":" + port;
        Debug.Log("XAIRClient starting at " + _serverUri);

        // "origin"
        if (enableDebug)
        {
            GameObject marker = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            marker.transform.localPosition = new Vector3(0.0f, 1.6f, 0.0f);
            marker.transform.localScale = new Vector3(0.02f, 0.02f, 0.02f);
        }

        // transparentMat = new Material(Shader.Find("Universal Render Pipeline/Lit"));
        // transparentMat.SetFloat("_Surface", 1); // 1 = Transparent, 0 = Opaque
        // transparentMat.SetFloat("_ZWrite", 0);  // Disable ZWrite for transparency
        // transparentMat.EnableKeyword("_SURFACE_TYPE_TRANSPARENT");
        // transparentMat.renderQueue = (int)UnityEngine.Rendering.RenderQueue.Transparent;
        // transparentMat.color = new Color(1f, 0f, 0f, 0.9f); // semi-transparent

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
            _mediaManager.GetImage(
                (cameraToWorldMatrix, instrinsics, distortion, imageBytes) =>
                {
                    if (imageBytes != null)
                    {
                        // send image to server
                        _serverCommunication.SendImage(imageBytes, cameraToWorldMatrix, instrinsics, distortion);
                    }
                },
                png: false
            );
        }
    }

    private void PlayDing()
    {
        if (dingSource != null)
        {
            dingSource.Play();
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
            if (msgType == "LLMReply")
            {
                var content = jsonObj["content"] as JsonObject;
                string llmReply = content["reply"].ToString();
                bool stepCompleted = bool.Parse(content["stepCompleted"].ToString());
                if (stepCompleted)
                {
                    PlayDing();
                }
                UpdateLLMResponseText(llmReply);
            }
            else
            if (msgType == "objectDetections")
            {
                // parse json
                uint imageWidth = uint.Parse(jsonObj["imageWidth"].ToString());
                uint imageHeight = uint.Parse(jsonObj["imageHeight"].ToString());
                Vector2 imageDimensions = new(imageWidth, imageHeight);

                Matrix4x4 cameraToWorldMatrix = StringToMatrix(jsonObj["extrinsics"].ToString());
                Matrix4x4 instrinsics = StringToMatrix(jsonObj["instrinsics"].ToString());

                Matrix4x4 distortion = StringToMatrix(jsonObj["distortion"].ToString());

                var content = jsonObj["content"] as JsonObject;

                var objectLabelsArray = content["labels"] as JsonArray;
                List<string> objectLabels = new();
                foreach (var label in objectLabelsArray)
                {
                    if (label is string labelStr)
                    {
                        objectLabels.Add(labelStr);
                    }
                }

                var objectCentersArray = content["centers"] as JsonArray;
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

                var objectConfidencesArray = content["confidences"] as JsonArray;
                List<float> objectConfidences = new();
                foreach (var confidence in objectConfidencesArray)
                {
                    Debug.Log("Confidence: " + confidence + " type: " + (confidence.GetType() == typeof(string)));
                    objectConfidences.Add(float.Parse(confidence.ToString()));
                }

                // destroy old text objects
                for (int i = 0; i < activeObjects.Count; i++)
                {
                    Destroy(activeObjects[i]);
                }
                activeObjects.Clear();

                // project object centers (image space) into world space
                for (int i = 0; i < objectCenters.Count; i++)
                {
                    string objectLabel = objectLabels[i];
                    Vector2 pixelCoords = objectCenters[i];
                    if (RaycastToMesh(pixelCoords, imageDimensions, cameraToWorldMatrix, instrinsics, distortion, out Vector3 hitPoint))
                    {
                        GameObject textObject = new("LabelText");
                        textObject.transform.localScale = new Vector3(0.125f, 0.125f, 0.125f);
                        textObject.transform.position = hitPoint + new Vector3(0.0f, -0.015f, 0.0f);

                        TextMesh textMesh = textObject.AddComponent<TextMesh>();
                        textMesh.text = $"{objectLabel}";
                        textMesh.font = Resources.GetBuiltinResource<Font>("Arial.ttf");
                        textMesh.fontSize = 90;
                        textMesh.characterSize = 0.01f;
                        textMesh.color = new Color(1f, 0f, 0f, 0.9f);
                        textMesh.alignment = TextAlignment.Center;
                        textMesh.anchor = TextAnchor.MiddleCenter;

                        textObject.transform.LookAt(Camera.main.transform);
                        textObject.transform.Rotate(0, 180, 0);
                        activeObjects.Add(textObject);

                        // make a transparent sphere at the hit point
                        GameObject marker = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                        marker.name = objectLabel;
                        marker.transform.localPosition = hitPoint;
                        marker.transform.localScale = new Vector3(0.0075f, 0.0075f, 0.0075f);
                        // marker.GetComponent<Renderer>().material = transparentMat;
                        activeObjects.Add(marker);

                        if (enableDebug)
                        {
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
            }
        }
    }

    private Vector2 NormalizePixel(Vector2 pixel, float fx, float fy, float cx, float cy)
    {
        // Normalize pixel
        float x = (pixel.x - cx) / fx;
        float y = (pixel.y - cy) / fy;

        return new Vector2(x, y);
    }

    private bool RaycastToMesh(Vector2 pixelCoords, Vector2 imageDimensions, Matrix4x4 cameraToWorldMatrix, Matrix4x4 instrinsics, Matrix4x4 distortion, out Vector3 hitPoint)
    {
        float fx = instrinsics.m00;
        float fy = instrinsics.m11;
        float cx = instrinsics.m02;
        float cy = instrinsics.m12;

        // Normalize pixel coordinates
        Vector2 undistorted = NormalizePixel(new Vector2(pixelCoords.x, pixelCoords.y), fx, fy, cx, cy);

        // // Convert to camera space direction
        Vector3 cameraSpacePoint = new(undistorted.x, undistorted.y, 1.0f);
        var rotation = cameraToWorldMatrix.rotation;
        Vector3 rayDirWorld = (rotation * cameraSpacePoint).normalized;
        Vector3 cameraPositionWorld = cameraToWorldMatrix.GetPosition();

        Ray ray = new(cameraPositionWorld, rayDirWorld);
        hitPoint = Vector3.zero;
        return false;
    }

    private void UpdateLLMResponseText(string text)
    {
        if (LLMResponseText != null)
        {
            LLMResponseText.text = text;
        }
    }
};
