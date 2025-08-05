/// <summary>
/// ServerCommunication manages the communication with the signaling server for WebRTC connections.
/// It handles login, sending and receiving SDP offers and answers, and managing ICE candidates.
/// </summary>
using SimpleJson;
using System;
using System.Text;
using System.Collections;
using Unity.WebRTC;
using UnityEngine;
using UnityEngine.Networking;
using System.Linq;

public class ServerCommunication : MonoBehaviour
{
    [System.Serializable]
    public class AnnotationData
    {
        public int id;
        public int[][] coordinates;  // Array of [x,y] coordinates
        public int[] center;         // [x, y] center point
        public BoundingBox bounding_box;
        public float[] pose_matrix;  // The original camera pose matrix
        public long timestamp;
        public bool processed;
    }
    [System.Serializable]
    public class BoundingBox
    {
        public int[] min;  // [x, y]
        public int[] max;  // [x, y]
        public int width;
        public int height;
    }

    [System.Serializable]
    public class AnnotationResponse
    {
        public string status;
        public AnnotationData annotation;  // Single annotation
        public AnnotationData[] annotations; // Multiple annotations
        public int count;
    }
    [System.Serializable]
    public class ProcessedAnnotations
    {
        public int[] processed_ids;
    }
    [System.Serializable]
    private class ImageData
    {
        public long timestamp;
        public float[] cameraToWorldMatrix;
        public float[] instrinsics;

        public float[] distortion;
        public string image;
    }

    // Action to notify when login to the server is complete
    public Action<bool> OnLoginAnswer;

    // Action to handle incoming SDP offers
    public Action<string> OnRemoteSDPOffer;

    // Action to notify when SDP offer has been sent to the server
    public Action OnSDPOfferSentInServer;

    // Action to handle checking of SDP answers from the server
    public Action<string> OnAnswerChecked;

    // Action to handle new remote ICE candidates
    public Action<string, string, int> OnNewRemoteICECandidate;

    // Action to notify when new annotation is received
    public Action<AnnotationData> OnNewAnnotationReceived;

    // Action to notify when annotation polling fails
    public Action<string> OnAnnotationError;

    // Manages concurrent web requests
    private ConcurrentWebRequestManager _webRequestManager = new ConcurrentWebRequestManager();

    // Server and participant information
    private string _serverIP = "";
    private string _serverUri = "";
    private string _localId = ""; // Local ID given by the server
    private string _remoteId = ""; // Remote participant ID

    private bool _isPollingAnnotations = false;
    private float _annotationPollInterval = 1.0f; // Poll every 1 second
    private Coroutine _annotationPollingCoroutine;

    private float[] MatrixToFloatArray(Matrix4x4 matrix)
    {
        return new float[]
        {
            matrix.m00, matrix.m01, matrix.m02, matrix.m03,
            matrix.m10, matrix.m11, matrix.m12, matrix.m13,
            matrix.m20, matrix.m21, matrix.m22, matrix.m23,
            matrix.m30, matrix.m31, matrix.m32, matrix.m33
        };
    }

    /// <summary>
    /// Log in to the server.
    /// </summary>
    public void Login()
    {
        try
        {
            Debug.Log("Sending POST LOGIN request");
            _webRequestManager.HttpPost(_serverUri + "/login", string.Empty, (AsyncOperation asyncOp) =>
            {
                if (asyncOp is not UnityWebRequestAsyncOperation webRequestAsyncOp)
                {
                    Debug.LogError("WebRequest is NULL");
                    OnLoginAnswer?.Invoke(false);
                    return;
                }

                Debug.Log(webRequestAsyncOp.webRequest.result.ToString());
                if (webRequestAsyncOp.webRequest.result != UnityWebRequest.Result.Success || string.IsNullOrEmpty(webRequestAsyncOp.webRequest.downloadHandler.text))
                {
                    Debug.LogError("Operation Error connecting to the server");
                    Debug.Log("Error connecting with the server, check if you are running the Python server locally.");
                    OnLoginAnswer?.Invoke(false);
                    return;
                }

                // Save ID given by the server
                _localId = webRequestAsyncOp.webRequest.downloadHandler.text;
                OnLoginAnswer?.Invoke(true);
            });
        }
        catch (UriFormatException)
        {
            Debug.LogError($"Bad URI: hostname \"{_serverUri}\" could not be parsed.");
        }
    }

    private void Update()
    {
        // Process web request manager queue
        _webRequestManager.UpdateWebRequests();
    }

    /// <summary>
    /// Check the server for any awaiting SDP offers.
    /// </summary>
    public void QueryOffers()
    {
        _webRequestManager.HttpGet(_serverUri + "/offers", (AsyncOperation asyncOp) =>
        {
            UnityWebRequestAsyncOperation webRequestAsyncOp = asyncOp as UnityWebRequestAsyncOperation;
            string offers = webRequestAsyncOp.webRequest.downloadHandler.text;
            if (ParseOffers(offers, out string remoteId, out string sdp))
            {
                _remoteId = remoteId;
                OnRemoteSDPOffer?.Invoke(sdp);
            }
            else
            {
                OnRemoteSDPOffer?.Invoke("");
            }
        });
    }

    private bool ParseOffers(string data, out string remoteId, out string sdp)
    {
        bool result = false;
        sdp = "";
        remoteId = "";

        if (data == "{}" || data == string.Empty)
        {
            return result;
        }

        SimpleJson.SimpleJson.TryDeserializeObject(data, out object obj);
        JsonObject jsonObj = (JsonObject)obj;
        foreach (var pair in jsonObj)
        {
            remoteId = pair.Key;
            JsonObject offerObj = (JsonObject)pair.Value;
            sdp = (string)offerObj["sdp"];
            result = true;
        }

        return result;
    }

    /// <summary>
    /// Initialize the server communication with the given server IP.
    /// </summary>
    /// <param name="serverIP">Server IP address.</param>
    public void Init(string serverIP)
    {
        _serverIP = serverIP;
        _serverUri = CreateServerURI(serverIP);
    }

    private string CreateServerURI(string serverAddress)
    {
        return serverAddress;
    }

    public string GetServerIP()
    {
        return _serverIP;
    }

    /// <summary>
    /// Send an SDP answer to the signaling server.
    /// </summary>
    /// <param name="answerSdp">The SDP answer.</param>
    public void SendAnswerToSignalServer(string answerSdp)
    {
        Debug.Log("Sending SDP answer to the server...");
        _webRequestManager.HttpPost(_serverUri + "/post_answer/" + _localId + "/" + _remoteId, FormatSdpOffer("answer", answerSdp));
    }

    public static string FormatSdpOffer(string offer, string sdp)
    {
        JsonObject jsonObj = new JsonObject
        {
            ["sdp"] = sdp,
            ["type"] = offer
        };
        return jsonObj.ToString();
    }

    /// <summary>
    /// Send an SDP offer to the signaling server.
    /// </summary>
    /// <param name="sdpOffer">The SDP offer.</param>
    public void SendOfferToSignalServer(string sdpOffer)
    {
        Debug.Log("Sending SDP offer to the server...");
        _webRequestManager.HttpPost(_serverUri + "/post_offer/" + _localId, FormatSdpOffer("offer", sdpOffer), (AsyncOperation ao) =>
        {
            OnSDPOfferSentInServer?.Invoke();
        });
    }

    /// <summary>
    /// Check the server for any SDP answers.
    /// </summary>
    public void CheckAnswers()
    {
        _webRequestManager.HttpGet(_serverUri + "/answer/" + _localId, (AsyncOperation asyncOp) =>
        {
            UnityWebRequestAsyncOperation webRequestAsyncOp = asyncOp as UnityWebRequestAsyncOperation;
            string response = webRequestAsyncOp.webRequest.downloadHandler.text;
            if (ParseAnswer(response, out string remoteId, out string remoteAnswer))
            {
                _remoteId = remoteId;
                OnAnswerChecked?.Invoke(remoteAnswer);
            }
            else
            {
                OnAnswerChecked?.Invoke("");
            }
        });
    }

    private bool ParseAnswer(string data, out string remoteId, out string sdp)
    {
        bool result = false;
        sdp = "";
        remoteId = "";

        if (data == "{}" || data == string.Empty)
        {
            return result;
        }

        SimpleJson.SimpleJson.TryDeserializeObject(data, out object obj);
        if (obj == null)
        {
            return false;
        }

        JsonObject jsonObj = (JsonObject)obj;
        if (jsonObj.ContainsKey("id") && jsonObj.ContainsKey("answer"))
        {
            remoteId = ((long)jsonObj["id"]).ToString();
            JsonObject answerObj = (JsonObject)jsonObj["answer"];
            sdp = (string)answerObj["sdp"];
            result = true;
        }

        return result;
    }

    /// <summary>
    /// Send an ICE candidate to the signaling server.
    /// </summary>
    /// <param name="candidate">The ICE candidate.</param>
    public void SendICECandidate(RTCIceCandidate candidate)
    {
        Debug.Log("Sending ICE candidate...");
        _webRequestManager.HttpPost(_serverUri + "/post_ice/" + _localId, FormatIceCandidate(candidate));
    }

    private string FormatIceCandidate(RTCIceCandidate iceCandidate)
    {
        JsonObject jsonObj = new JsonObject
        {
            ["candidate"] = iceCandidate.Candidate,
            ["sdpMLineIndex"] = iceCandidate.SdpMLineIndex,
            ["sdpMid"] = iceCandidate.SdpMid
        };
        return jsonObj.ToString();
    }

    /// <summary>
    /// Check the server for any remote ICE candidates.
    /// </summary>
    public void CheckRemoteIce()
    {
        if (string.IsNullOrEmpty(_remoteId))
        {
            Debug.LogError("Remote ID is null when checking remote ICEs");
            return;
        }

        _webRequestManager.HttpPost(_serverUri + "/consume_ices/" + _remoteId, "", (AsyncOperation asyncOp) =>
        {
            Debug.Log("Consuming ICE candidates");

            UnityWebRequestAsyncOperation webRequestAsyncOp = asyncOp as UnityWebRequestAsyncOperation;
            string iceCandidates = webRequestAsyncOp.webRequest.downloadHandler.text;

            // Parses all the ice candidates
            JsonObject jsonObjects = (JsonObject)SimpleJson.SimpleJson.DeserializeObject(iceCandidates);
            JsonArray jsonArray = (JsonArray)jsonObjects[0];

            foreach (JsonObject jsonObj in jsonArray.Cast<JsonObject>())
            {
                OnNewRemoteICECandidate?.Invoke((string)jsonObj["candidate"], (string)jsonObj["sdpMid"], Convert.ToInt32(jsonObj["sdpMLineIndex"]));
            }
        });
    }

    public void SendImage(byte[] image, Matrix4x4 cameraToWorldMatrix, Matrix4x4 instrinsics, Matrix4x4 distortion)
    {
        StartCoroutine(SendImageCoroutine(image, cameraToWorldMatrix, instrinsics, distortion));
    }

    private IEnumerator SendImageCoroutine(byte[] image, Matrix4x4 cameraToWorldMatrix, Matrix4x4 instrinsics, Matrix4x4 distortion)
    {
        string base64Image = Convert.ToBase64String(image);

        // Convert matrices to float arrays
        float[] camToWorldArray = MatrixToFloatArray(cameraToWorldMatrix);
        float[] projArray = MatrixToFloatArray(instrinsics);
        float[] distArray = MatrixToFloatArray(distortion);

        DateTime epochStart = new(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc);
        long currTime = (long)(DateTime.UtcNow - epochStart).TotalMilliseconds;
        var imageData = new ImageData
        {
            timestamp = currTime,
            image = base64Image,
            cameraToWorldMatrix = camToWorldArray,
            instrinsics = projArray,
            distortion = distArray
        };

        string jsonPayload = JsonUtility.ToJson(imageData);
        byte[] postData = Encoding.UTF8.GetBytes(jsonPayload);

        using UnityWebRequest request = new(_serverUri + "/post_image/" + _localId, "POST");
        request.uploadHandler = new UploadHandlerRaw(postData);
        request.downloadHandler = new DownloadHandlerBuffer();
        request.SetRequestHeader("Content-Type", "application/json");

        request.certificateHandler = new AcceptAnyCertificate();
        request.disposeCertificateHandlerOnDispose = true;

        yield return request.SendWebRequest();

        if (request.result != UnityWebRequest.Result.Success)
        {
            Debug.LogError("Error sending image: " + request.error);
        }
    }

    /// <summary>
    /// Start polling the server for new annotations
    /// </summary>
    /// <param name="pollInterval">How often to check for annotations in seconds</param>
    public void StartAnnotationPolling(float pollInterval = 1.0f)
    {
        if (_isPollingAnnotations)
        {
            Debug.LogWarning("Annotation polling is already running");
            return;
        }

        _annotationPollInterval = pollInterval;
        _isPollingAnnotations = true;
        _annotationPollingCoroutine = StartCoroutine(PollForAnnotations());

        Debug.Log($"🎯 Started annotation polling every {pollInterval} seconds");
    }

    /// <summary>
    /// Stop polling for annotations
    /// </summary>
    public void StopAnnotationPolling()
    {
        _isPollingAnnotations = false;

        if (_annotationPollingCoroutine != null)
        {
            StopCoroutine(_annotationPollingCoroutine);
            _annotationPollingCoroutine = null;
        }

        Debug.Log("🛑 Stopped annotation polling");
    }

    /// <summary>
    /// Get the latest annotation from the server (single request)
    /// </summary>
    public void GetLatestAnnotation()
    {
        StartCoroutine(GetLatestAnnotationCoroutine());
    }

    /// <summary>
    /// Mark annotations as processed by Unity
    /// </summary>
    /// <param name="annotationIds">Array of annotation IDs that have been processed</param>
    public void MarkAnnotationsAsProcessed(int[] annotationIds)
    {
        StartCoroutine(MarkAnnotationsProcessedCoroutine(annotationIds));
    }

    private IEnumerator PollForAnnotations()
    {
        while (_isPollingAnnotations)
        {
            yield return new WaitForSeconds(_annotationPollInterval);
            yield return StartCoroutine(GetLatestAnnotationCoroutine());
        }
    }

    private IEnumerator GetLatestAnnotationCoroutine()
    {
        string url = _serverUri + "/get-latest-annotation";

        using (UnityWebRequest request = UnityWebRequest.Get(url))
        {
            // Add certificate handler for HTTPS
            request.certificateHandler = new AcceptAnyCertificate();
            request.disposeCertificateHandlerOnDispose = true;

            yield return request.SendWebRequest();

            if (request.result == UnityWebRequest.Result.Success)
            {
                try
                {
                    string responseText = request.downloadHandler.text;
                    Debug.Log($"📥 Annotation response: {responseText}");

                    AnnotationResponse response = JsonUtility.FromJson<AnnotationResponse>(responseText);

                    if (response.status == "success" && response.annotation != null)
                    {
                        Debug.Log($"🎯 New annotation received: ID {response.annotation.id}");
                        Debug.Log($"📊 Center: ({response.annotation.center[0]}, {response.annotation.center[1]})");
                        Debug.Log($"📐 Pose matrix elements: {response.annotation.pose_matrix?.Length ?? 0}");

                        // Notify listeners (your ARPlaceCube script can subscribe to this)
                        OnNewAnnotationReceived?.Invoke(response.annotation);
                    }
                    else if (response.status == "no_annotations")
                    {
                        // No new annotations - this is normal, don't spam logs
                    }
                    else if (response.status == "already_processed")
                    {
                        // Already processed - this is normal
                    }
                    else
                    {
                        Debug.LogWarning($"⚠️ Unexpected annotation response status: {response.status}");
                    }
                }
                catch (System.Exception e)
                {
                    Debug.LogError($"❌ Error parsing annotation response: {e.Message}");
                    OnAnnotationError?.Invoke($"Failed to parse annotation: {e.Message}");
                }
            }
            else
            {
                Debug.LogWarning($"⚠️ Failed to get annotations: {request.error}");
                OnAnnotationError?.Invoke($"Network error: {request.error}");
            }
        }
    }

    private IEnumerator MarkAnnotationsProcessedCoroutine(int[] annotationIds)
    {
        string url = _serverUri + "/mark-annotations-processed";

        ProcessedAnnotations payload = new ProcessedAnnotations
        {
            processed_ids = annotationIds
        };

        string jsonPayload = JsonUtility.ToJson(payload);
        byte[] postData = Encoding.UTF8.GetBytes(jsonPayload);

        using (UnityWebRequest request = new UnityWebRequest(url, "POST"))
        {
            request.uploadHandler = new UploadHandlerRaw(postData);
            request.downloadHandler = new DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "application/json");

            request.certificateHandler = new AcceptAnyCertificate();
            request.disposeCertificateHandlerOnDispose = true;

            yield return request.SendWebRequest();

            if (request.result == UnityWebRequest.Result.Success)
            {
                Debug.Log($"✅ Marked {annotationIds.Length} annotations as processed");
            }
            else
            {
                Debug.LogWarning($"⚠️ Failed to mark annotations as processed: {request.error}");
            }
        }
    }

    /// <summary>
    /// Log out and disconnect from the server.
    /// </summary>
    public void Disconnect()
    {
        StopAnnotationPolling();
        _webRequestManager.HttpPost(_serverUri + "/logout/" + _localId, string.Empty);
    }


}
