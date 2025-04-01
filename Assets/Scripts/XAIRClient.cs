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
    private GameObject prefab;

    [SerializeField]
    private TextMeshProUGUI LLMResponseText = null;

    private readonly bool debug = false;

    private string _serverUri;
    private bool connected = false;

    private float _sendImagetimer = 0.0f;
    [SerializeField]
    private float _sendImageFreqHz = 1.0f;

    public Action<bool, string> OnWebRTCConnectionChanged;

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
        if (debug) {
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
                                        out Matrix4x4 distortion,
                                        true);
            if (imageBytes != null)
            {
                // offset y coordinate by 1.6
                cameraToWorldMatrix.m13 += 1.6f;
                // send image to server
                _serverCommunication.SendImage(imageBytes, cameraToWorldMatrix, instrinsics, distortion);
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

                string llm_reply = jsonObj["content"].ToString();

                uint imageWidth = uint.Parse(jsonObj["imageWidth"].ToString());
                uint imageHeight = uint.Parse(jsonObj["imageHeight"].ToString());
                Vector2 imageDimensions = new(imageWidth, imageHeight);

                Matrix4x4 cameraToWorldMatrix = StringToMatrix(jsonObj["extrinsics"].ToString());
                Matrix4x4 instrinsics = StringToMatrix(jsonObj["instrinsics"].ToString());
                Matrix4x4 distortion = StringToMatrix(jsonObj["distortion"].ToString());

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

                Vector3 cameraPositionWorld = cameraToWorldMatrix.GetColumn(3);

                // project object centers (image space) into world space
                for (int i = 0; i < objectCenters.Count; i++)
                {
                    Vector2 pixelCoords = objectCenters[i];
                    if (RaycastToMesh(pixelCoords, imageDimensions, cameraToWorldMatrix, instrinsics, distortion, out Vector3 hitPoint))
                    {
                        prefab.transform.localScale = new Vector3(0.02f, 0.02f, 0.02f);
                        Instantiate(prefab, hitPoint, Quaternion.identity);

                        if (debug) {
                            LineRenderer lr = new GameObject("RayLine").AddComponent<LineRenderer>();
                            lr.positionCount = 2;
                            lr.SetPosition(0, cameraPositionWorld);
                            lr.SetPosition(1, hitPoint);
                            lr.startWidth = lr.endWidth = 0.01f;
                        }
                    }
                    else if (debug)
                    {
                        Debug.LogWarning("Raycast failed to hit mesh.");
                    }
                }

                UpdateLLMResponseText(llm_reply);
            }
        }
    }

    private bool RaycastToMesh(Vector2 pixelCoords, Vector2 imageDimensions, Matrix4x4 cameraToWorldMatrix, Matrix4x4 instrinsics, Matrix4x4 distortion, out Vector3 hitPoint)
    {
        // image space to ndc (normalized device coordinates) (-1 to 1)
        float fx = instrinsics.m00;
        float fy = instrinsics.m11;
        float cx = instrinsics.m02;
        float cy = instrinsics.m12;

        float x = (pixelCoords.x - cx) / fx;
        float y = (pixelCoords.y - (imageDimensions.y - cy)) / fy; // invert y axis

        float k1 = distortion.m00;
        float k2 = distortion.m01;
        float p1 = distortion.m02;
        float p2 = distortion.m03;
        float k3 = distortion.m10;

        // undistort point
        Vector2 undistorted = UndistortPoint(new Vector2(x, y), k1, k2, p1, p2, k3);
        Vector3 cameraSpacePoint = new(undistorted.x, undistorted.y, 1.0f);

        // project to world space
        Vector3 rayDirWorld = cameraToWorldMatrix.MultiplyVector(cameraSpacePoint);

        // raycast from camera to object
        Vector3 cameraPositionWorld = cameraToWorldMatrix.GetColumn(3);
        Ray ray = new(cameraPositionWorld, rayDirWorld);
        return _meshManager.RayCastToMesh(ray, out hitPoint);
    }
    private Vector2 UndistortPoint(Vector2 distorted, float k1, float k2, float p1, float p2, float k3)
    {
        Vector2 undistorted = distorted;
        for (int i = 0; i < 5; i++) // usually 5 iterations is enough
        {
            float x = undistorted.x;
            float y = undistorted.y;
            float r2 = x * x + y * y;
            float radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
            float xTangential = 2 * p1 * x * y + p2 * (r2 + 2 * x * x);
            float yTangential = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y;
            undistorted.x = (distorted.x - xTangential) / radial;
            undistorted.y = (distorted.y - yTangential) / radial;
        }
        return undistorted;
    }

    private void UpdateLLMResponseText(string text)
    {
        if (LLMResponseText != null)
        {
            LLMResponseText.text = text;
        }
    }
};
