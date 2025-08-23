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
    /// <summary>
/// Log in to the server.
/// </summary>
    public void Login()
    {
        StartCoroutine(LoginCoroutine());
    }

    private IEnumerator LoginCoroutine()
    {
        Debug.Log("Sending POST LOGIN request");
        
        using (UnityWebRequest request = new UnityWebRequest(_serverUri + "/login", "POST"))
        {
            request.uploadHandler = new UploadHandlerRaw(System.Text.Encoding.UTF8.GetBytes(""));
            request.downloadHandler = new DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "application/json");
            
            // Add certificate handler for HTTPS
            request.certificateHandler = new AcceptAnyCertificate();
            request.disposeCertificateHandlerOnDispose = true;
            request.timeout = 10; // 10 second timeout
            
            yield return request.SendWebRequest();
            
            Debug.Log($"Login request result: {request.result}");
            Debug.Log($"Login response code: {request.responseCode}");
            Debug.Log($"Login response: {request.downloadHandler.text}");
            
            if (request.result == UnityWebRequest.Result.Success && !string.IsNullOrEmpty(request.downloadHandler.text))
            {
                _localId = request.downloadHandler.text;
                Debug.Log($"✅ Login successful! Received ID: {_localId}");
                OnLoginAnswer?.Invoke(true);
            }
            else
            {
                Debug.LogError($"❌ Login failed: {request.error}");
                Debug.LogError($"Response: {request.downloadHandler.text}");
                OnLoginAnswer?.Invoke(false);
            }
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

// /// <summary>
// /// Simple WebSocket client using Unity's UnityWebRequest for audio streaming
// /// No external packages required - uses HTTP POST for audio transmission
// /// </summary>
// using System;
// using System.Text;
// using System.Collections;
// using UnityEngine;
// using UnityEngine.Networking;
// using SimpleJson;
// using System.Linq;

// public class ServerCommunication : MonoBehaviour
// {
//     [System.Serializable]
//     public class AnnotationData
//     {
//         public int id;
//         public int[][] coordinates;
//         public int[] center;
//         public BoundingBox bounding_box;
//         public float[] pose_matrix;
//         public long timestamp;
//         public bool processed;
//     }

//     [System.Serializable]
//     public class BoundingBox
//     {
//         public int[] min;
//         public int[] max;
//         public int width;
//         public int height;
//     }

//     [System.Serializable]
//     public class AnnotationResponse
//     {
//         public string status;
//         public AnnotationData annotation;
//         public AnnotationData[] annotations;
//         public int count;
//     }

//     [System.Serializable]
//     public class ProcessedAnnotations
//     {
//         public int[] processed_ids;
//     }

//     [System.Serializable]
//     private class ImageData
//     {
//         public long timestamp;
//         public float[] cameraToWorldMatrix;
//         public float[] instrinsics;
//         public float[] distortion;
//         public string image;
//     }

//     // ===== AUDIO DATA CLASSES =====
//     [System.Serializable]
//     public class AudioMessage
//     {
//         public string type;
//         public int[] audioData;
//         public int sampleRate;
//         public long timestamp;
//         public string source;
//     }

//     [System.Serializable]
//     public class AudioFromWebMessage
//     {
//         public string type;
//         public int[] audioData;
//         public int sampleRate;
//         public long timestamp;
//     }

//     // Existing actions
//     public Action<bool> OnLoginAnswer;
//     public Action<string> OnRemoteSDPOffer;
//     public Action OnSDPOfferSentInServer;
//     public Action<string> OnAnswerChecked;
//     public Action<string, string, int> OnNewRemoteICECandidate;
//     public Action<AnnotationData> OnNewAnnotationReceived;
//     public Action<string> OnAnnotationError;

//     // ===== AUDIO ACTIONS =====
//     public Action<int[], int> OnAudioReceivedFromWeb;  // Audio data and sample rate
//     public Action<bool> OnAudioConnectionStatus;       // Connection status
//     public Action<string> OnAudioError;                // Audio errors

//     private ConcurrentWebRequestManager _webRequestManager = new ConcurrentWebRequestManager();

//     // Server and participant information
//     private string _serverIP = "";
//     private string _serverUri = "";
//     private string _localId = "";
//     private string _remoteId = "";

//     // Annotation polling
//     private bool _isPollingAnnotations = false;
//     private float _annotationPollInterval = 1.0f;
//     private Coroutine _annotationPollingCoroutine;

//     // ===== AUDIO POLLING VARIABLES =====
//     private bool _isPollingAudio = false;
//     private float _audioPollInterval = 0.1f; // Poll every 100ms for audio
//     private Coroutine _audioPollingCoroutine;

//     // DataChannel reference for receiving audio messages
//     private Unity.WebRTC.RTCDataChannel _dataChannel;

//     private float[] MatrixToFloatArray(Matrix4x4 matrix)
//     {
//         return new float[]
//         {
//             matrix.m00, matrix.m01, matrix.m02, matrix.m03,
//             matrix.m10, matrix.m11, matrix.m12, matrix.m13,
//             matrix.m20, matrix.m21, matrix.m22, matrix.m23,
//             matrix.m30, matrix.m31, matrix.m32, matrix.m33
//         };
//     }

//     // ===== DATACHANNEL SETUP =====
//     /// <summary>
//     /// Set the DataChannel reference for audio communication
//     /// Call this from your WebRTC script when DataChannel is created
//     /// </summary>
//     public void SetDataChannel(Unity.WebRTC.RTCDataChannel dataChannel)
//     {
//         _dataChannel = dataChannel;
//         Debug.Log("🎤 DataChannel reference set for audio communication");
//     }

//     /// <summary>
//     /// Process incoming DataChannel messages (call this from WebRTC message handler)
//     /// </summary>
//     public void ProcessDataChannelMessage(string message)
//     {
//         try
//         {
//             // Parse message to check if it's audio data
//             var messageObj = JsonUtility.FromJson<AudioFromWebMessage>(message);
            
//             if (messageObj.type == "live_audio_from_web")
//             {
//                 Debug.Log($"🎤 Received audio from web: {messageObj.audioData.Length} samples");
//                 OnAudioReceivedFromWeb?.Invoke(messageObj.audioData, messageObj.sampleRate);
//             }
//         }
//         catch (Exception e)
//         {
//             // Not audio data or invalid JSON - ignore silently
//             Debug.Log($"DataChannel message not audio: {e.Message}");
//         }
//     }

//     // ===== LOGIN WITH AUDIO POLLING =====
//     public void Login()
//     {
//         StartCoroutine(LoginCoroutine());
//     }

//     private IEnumerator LoginCoroutine()
//     {
//         Debug.Log("Sending POST LOGIN request");
        
//         using (UnityWebRequest request = new UnityWebRequest(_serverUri + "/login", "POST"))
//         {
//             request.uploadHandler = new UploadHandlerRaw(System.Text.Encoding.UTF8.GetBytes(""));
//             request.downloadHandler = new DownloadHandlerBuffer();
//             request.SetRequestHeader("Content-Type", "application/json");
            
//             request.certificateHandler = new AcceptAnyCertificate();
//             request.disposeCertificateHandlerOnDispose = true;
//             request.timeout = 10;
            
//             yield return request.SendWebRequest();
            
//             Debug.Log($"Login request result: {request.result}");
//             Debug.Log($"Login response code: {request.responseCode}");
//             Debug.Log($"Login response: {request.downloadHandler.text}");
            
//             if (request.result == UnityWebRequest.Result.Success && !string.IsNullOrEmpty(request.downloadHandler.text))
//             {
//                 _localId = request.downloadHandler.text;
//                 Debug.Log($"✅ Login successful! Received ID: {_localId}");
//                 OnLoginAnswer?.Invoke(true);
                
//                 // Start audio polling for receiving audio from web
//                 StartAudioPolling();
//             }
//             else
//             {
//                 Debug.LogError($"❌ Login failed: {request.error}");
//                 Debug.LogError($"Response: {request.downloadHandler.text}");
//                 OnLoginAnswer?.Invoke(false);
//             }
//         }
//     }

//     // ===== AUDIO POLLING METHODS =====
//     /// <summary>
//     /// Start polling the server for audio from web clients
//     /// Alternative to WebSocket - uses HTTP GET requests
//     /// </summary>
//     public void StartAudioPolling(float pollInterval = 0.1f)
//     {
//         if (_isPollingAudio)
//         {
//             Debug.LogWarning("Audio polling is already running");
//             return;
//         }

//         _audioPollInterval = pollInterval;
//         _isPollingAudio = true;
//         _audioPollingCoroutine = StartCoroutine(PollForAudio());

//         Debug.Log($"🎤 Started audio polling every {pollInterval} seconds");
//         OnAudioConnectionStatus?.Invoke(true);
//     }

//     /// <summary>
//     /// Stop polling for audio
//     /// </summary>
//     public void StopAudioPolling()
//     {
//         _isPollingAudio = false;

//         if (_audioPollingCoroutine != null)
//         {
//             StopCoroutine(_audioPollingCoroutine);
//             _audioPollingCoroutine = null;
//         }

//         Debug.Log("🔇 Stopped audio polling");
//         OnAudioConnectionStatus?.Invoke(false);
//     }

//     private IEnumerator PollForAudio()
//     {
//         while (_isPollingAudio)
//         {
//             yield return StartCoroutine(GetLatestAudioCoroutine());
//             yield return new WaitForSeconds(_audioPollInterval);
//         }
//     }

//     private IEnumerator GetLatestAudioCoroutine()
//     {
//         string url = _serverUri + "/live-audio-buffer";

//         using (UnityWebRequest request = UnityWebRequest.Get(url))
//         {
//             request.certificateHandler = new AcceptAnyCertificate();
//             request.disposeCertificateHandlerOnDispose = true;
//             request.timeout = 2; // Short timeout for audio

//             yield return request.SendWebRequest();

//             if (request.result == UnityWebRequest.Result.Success)
//             {
//                 try
//                 {
//                     string responseText = request.downloadHandler.text;
                    
//                     // Parse audio buffer response
//                     var response = JsonUtility.FromJson<AudioBufferResponse>(responseText);

//                     if (response.status == "success" && response.audio_items != null)
//                     {
//                         foreach (var audioItem in response.audio_items)
//                         {
//                             if (audioItem.source == "web" && audioItem.data != null)
//                             {
//                                 Debug.Log($"🎤 Received audio buffer: {audioItem.data.Length} samples");
//                                 OnAudioReceivedFromWeb?.Invoke(audioItem.data, audioItem.sample_rate);
//                             }
//                         }
//                     }
//                 }
//                 catch (System.Exception e)
//                 {
//                     // Audio polling errors are common when no audio is available
//                     // Don't spam the console
//                     Debug.Log($"Audio polling: {e.Message}");
//                 }
//             }
//             else if (request.responseCode != 404) // Don't log 404s as they're normal
//             {
//                 Debug.LogWarning($"⚠️ Audio polling failed: {request.error}");
//                 OnAudioError?.Invoke($"Audio polling error: {request.error}");
//             }
//         }
//     }

//     [System.Serializable]
//     public class AudioBufferResponse
//     {
//         public string status;
//         public AudioBufferItem[] audio_items;
//         public int buffer_size;
//     }

//     [System.Serializable]
//     public class AudioBufferItem
//     {
//         public string source;
//         public int[] data;
//         public int sample_rate;
//         public float timestamp;
//         public string client_ip;
//     }

//     /// <summary>
//     /// Send audio data to web clients via HTTP POST
//     /// This method works without WebSocket
//     /// </summary>
//     public void SendAudioToWeb(int[] audioSamples, int sampleRate)
//     {
//         StartCoroutine(SendAudioToWebCoroutine(audioSamples, sampleRate));
//     }

//     private IEnumerator SendAudioToWebCoroutine(int[] audioSamples, int sampleRate)
//     {
//         string url = _serverUri + "/unity-send-audio";
        
//         AudioMessage audioMessage = new AudioMessage
//         {
//             audioData = audioSamples,
//             sampleRate = sampleRate,
//             timestamp = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()
//         };
        
//         string jsonPayload = JsonUtility.ToJson(audioMessage);
//         byte[] postData = Encoding.UTF8.GetBytes(jsonPayload);
        
//         using (UnityWebRequest request = new UnityWebRequest(url, "POST"))
//         {
//             request.uploadHandler = new UploadHandlerRaw(postData);
//             request.downloadHandler = new DownloadHandlerBuffer();
//             request.SetRequestHeader("Content-Type", "application/json");
            
//             request.certificateHandler = new AcceptAnyCertificate();
//             request.disposeCertificateHandlerOnDispose = true;
            
//             yield return request.SendWebRequest();
            
//             if (request.result == UnityWebRequest.Result.Success)
//             {
//                 Debug.Log($"✅ Sent {audioSamples.Length} audio samples to web clients");
//             }
//             else
//             {
//                 Debug.LogError($"❌ Failed to send audio to web: {request.error}");
//                 OnAudioError?.Invoke($"Failed to send audio: {request.error}");
//             }
//         }
//     }

//     /// <summary>
//     /// Send audio data to web clients via DataChannel (faster, real-time)
//     /// </summary>
//     public void SendAudioViaDataChannel(int[] audioSamples, int sampleRate)
//     {
//         if (_dataChannel == null)
//         {
//             Debug.LogWarning("⚠️ DataChannel not set. Call SetDataChannel() first.");
//             return;
//         }
        
//         try
//         {
//             AudioMessage audioMessage = new AudioMessage
//             {
//                 type = "audio_from_unity",
//                 audioData = audioSamples,
//                 sampleRate = sampleRate,
//                 timestamp = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds(),
//                 source = "unity"
//             };
            
//             string jsonMessage = JsonUtility.ToJson(audioMessage);
//             _dataChannel.Send(jsonMessage);
            
//             Debug.Log($"📤 Sent {audioSamples.Length} audio samples via DataChannel");
//         }
//         catch (Exception e)
//         {
//             Debug.LogError($"❌ Failed to send audio via DataChannel: {e.Message}");
//             OnAudioError?.Invoke($"DataChannel send failed: {e.Message}");
//         }
//     }

//     /// <summary>
//     /// Check if audio system is connected (polling active)
//     /// </summary>
//     public bool IsAudioConnected()
//     {
//         return _isPollingAudio;
//     }

//     /// <summary>
//     /// Disconnect audio system
//     /// </summary>
//     public void DisconnectAudio()
//     {
//         StopAudioPolling();
//     }

//     // ===== UPDATE METHOD =====
//     private void Update()
//     {
//         _webRequestManager.UpdateWebRequests();
//     }

//     // ===== ALL OTHER EXISTING METHODS REMAIN THE SAME =====
//     public void QueryOffers()
//     {
//         _webRequestManager.HttpGet(_serverUri + "/offers", (AsyncOperation asyncOp) =>
//         {
//             UnityWebRequestAsyncOperation webRequestAsyncOp = asyncOp as UnityWebRequestAsyncOperation;
//             string offers = webRequestAsyncOp.webRequest.downloadHandler.text;
//             if (ParseOffers(offers, out string remoteId, out string sdp))
//             {
//                 _remoteId = remoteId;
//                 OnRemoteSDPOffer?.Invoke(sdp);
//             }
//             else
//             {
//                 OnRemoteSDPOffer?.Invoke("");
//             }
//         });
//     }

//     private bool ParseOffers(string data, out string remoteId, out string sdp)
//     {
//         bool result = false;
//         sdp = "";
//         remoteId = "";

//         if (data == "{}" || data == string.Empty)
//         {
//             return result;
//         }

//         SimpleJson.SimpleJson.TryDeserializeObject(data, out object obj);
//         JsonObject jsonObj = (JsonObject)obj;
//         foreach (var pair in jsonObj)
//         {
//             remoteId = pair.Key;
//             JsonObject offerObj = (JsonObject)pair.Value;
//             sdp = (string)offerObj["sdp"];
//             result = true;
//         }

//         return result;
//     }

//     public void Init(string serverIP)
//     {
//         _serverIP = serverIP;
//         _serverUri = CreateServerURI(serverIP);
//     }

//     private string CreateServerURI(string serverAddress)
//     {
//         return serverAddress;
//     }

//     public string GetServerIP()
//     {
//         return _serverIP;
//     }

//     public void SendAnswerToSignalServer(string answerSdp)
//     {
//         Debug.Log("Sending SDP answer to the server...");
//         _webRequestManager.HttpPost(_serverUri + "/post_answer/" + _localId + "/" + _remoteId, FormatSdpOffer("answer", answerSdp));
//     }

//     public static string FormatSdpOffer(string offer, string sdp)
//     {
//         JsonObject jsonObj = new JsonObject
//         {
//             ["sdp"] = sdp,
//             ["type"] = offer
//         };
//         return jsonObj.ToString();
//     }

//     public void SendOfferToSignalServer(string sdpOffer)
//     {
//         Debug.Log("Sending SDP offer to the server...");
//         _webRequestManager.HttpPost(_serverUri + "/post_offer/" + _localId, FormatSdpOffer("offer", sdpOffer), (AsyncOperation ao) =>
//         {
//             OnSDPOfferSentInServer?.Invoke();
//         });
//     }

//     public void CheckAnswers()
//     {
//         _webRequestManager.HttpGet(_serverUri + "/answer/" + _localId, (AsyncOperation asyncOp) =>
//         {
//             UnityWebRequestAsyncOperation webRequestAsyncOp = asyncOp as UnityWebRequestAsyncOperation;
//             string response = webRequestAsyncOp.webRequest.downloadHandler.text;
//             if (ParseAnswer(response, out string remoteId, out string remoteAnswer))
//             {
//                 _remoteId = remoteId;
//                 OnAnswerChecked?.Invoke(remoteAnswer);
//             }
//             else
//             {
//                 OnAnswerChecked?.Invoke("");
//             }
//         });
//     }

//     private bool ParseAnswer(string data, out string remoteId, out string sdp)
//     {
//         bool result = false;
//         sdp = "";
//         remoteId = "";

//         if (data == "{}" || data == string.Empty)
//         {
//             return result;
//         }

//         SimpleJson.SimpleJson.TryDeserializeObject(data, out object obj);
//         if (obj == null)
//         {
//             return false;
//         }

//         JsonObject jsonObj = (JsonObject)obj;
//         if (jsonObj.ContainsKey("id") && jsonObj.ContainsKey("answer"))
//         {
//             remoteId = ((long)jsonObj["id"]).ToString();
//             JsonObject answerObj = (JsonObject)jsonObj["answer"];
//             sdp = (string)answerObj["sdp"];
//             result = true;
//         }

//         return result;
//     }

//     public void SendICECandidate(Unity.WebRTC.RTCIceCandidate candidate)
//     {
//         Debug.Log("Sending ICE candidate...");
//         _webRequestManager.HttpPost(_serverUri + "/post_ice/" + _localId, FormatIceCandidate(candidate));
//     }

//     private string FormatIceCandidate(Unity.WebRTC.RTCIceCandidate iceCandidate)
//     {
//         JsonObject jsonObj = new JsonObject
//         {
//             ["candidate"] = iceCandidate.Candidate,
//             ["sdpMLineIndex"] = iceCandidate.SdpMLineIndex,
//             ["sdpMid"] = iceCandidate.SdpMid
//         };
//         return jsonObj.ToString();
//     }

//     public void CheckRemoteIce()
//     {
//         if (string.IsNullOrEmpty(_remoteId))
//         {
//             Debug.LogError("Remote ID is null when checking remote ICEs");
//             return;
//         }

//         _webRequestManager.HttpPost(_serverUri + "/consume_ices/" + _remoteId, "", (AsyncOperation asyncOp) =>
//         {
//             Debug.Log("Consuming ICE candidates");

//             UnityWebRequestAsyncOperation webRequestAsyncOp = asyncOp as UnityWebRequestAsyncOperation;
//             string iceCandidates = webRequestAsyncOp.webRequest.downloadHandler.text;

//             JsonObject jsonObjects = (JsonObject)SimpleJson.SimpleJson.DeserializeObject(iceCandidates);
//             JsonArray jsonArray = (JsonArray)jsonObjects[0];

//             foreach (JsonObject jsonObj in jsonArray.Cast<JsonObject>())
//             {
//                 OnNewRemoteICECandidate?.Invoke((string)jsonObj["candidate"], (string)jsonObj["sdpMid"], Convert.ToInt32(jsonObj["sdpMLineIndex"]));
//             }
//         });
//     }

//     public void SendImage(byte[] image, Matrix4x4 cameraToWorldMatrix, Matrix4x4 instrinsics, Matrix4x4 distortion)
//     {
//         StartCoroutine(SendImageCoroutine(image, cameraToWorldMatrix, instrinsics, distortion));
//     }

//     private IEnumerator SendImageCoroutine(byte[] image, Matrix4x4 cameraToWorldMatrix, Matrix4x4 instrinsics, Matrix4x4 distortion)
//     {
//         string base64Image = Convert.ToBase64String(image);

//         float[] camToWorldArray = MatrixToFloatArray(cameraToWorldMatrix);
//         float[] projArray = MatrixToFloatArray(instrinsics);
//         float[] distArray = MatrixToFloatArray(distortion);

//         DateTime epochStart = new(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc);
//         long currTime = (long)(DateTime.UtcNow - epochStart).TotalMilliseconds;
//         var imageData = new ImageData
//         {
//             timestamp = currTime,
//             image = base64Image,
//             cameraToWorldMatrix = camToWorldArray,
//             instrinsics = projArray,
//             distortion = distArray
//         };

//         string jsonPayload = JsonUtility.ToJson(imageData);
//         byte[] postData = Encoding.UTF8.GetBytes(jsonPayload);

//         using UnityWebRequest request = new(_serverUri + "/post_image/" + _localId, "POST");
//         request.uploadHandler = new UploadHandlerRaw(postData);
//         request.downloadHandler = new DownloadHandlerBuffer();
//         request.SetRequestHeader("Content-Type", "application/json");

//         request.certificateHandler = new AcceptAnyCertificate();
//         request.disposeCertificateHandlerOnDispose = true;

//         yield return request.SendWebRequest();

//         if (request.result != UnityWebRequest.Result.Success)
//         {
//             Debug.LogError("Error sending image: " + request.error);
//         }
//     }

//     // ===== ANNOTATION METHODS =====
//     public void StartAnnotationPolling(float pollInterval = 1.0f)
//     {
//         if (_isPollingAnnotations)
//         {
//             Debug.LogWarning("Annotation polling is already running");
//             return;
//         }

//         _annotationPollInterval = pollInterval;
//         _isPollingAnnotations = true;
//         _annotationPollingCoroutine = StartCoroutine(PollForAnnotations());

//         Debug.Log($"🎯 Started annotation polling every {pollInterval} seconds");
//     }

//     public void StopAnnotationPolling()
//     {
//         _isPollingAnnotations = false;

//         if (_annotationPollingCoroutine != null)
//         {
//             StopCoroutine(_annotationPollingCoroutine);
//             _annotationPollingCoroutine = null;
//         }

//         Debug.Log("🛑 Stopped annotation polling");
//     }

//     public void GetLatestAnnotation()
//     {
//         StartCoroutine(GetLatestAnnotationCoroutine());
//     }

//     public void MarkAnnotationsAsProcessed(int[] annotationIds)
//     {
//         StartCoroutine(MarkAnnotationsProcessedCoroutine(annotationIds));
//     }

//     private IEnumerator PollForAnnotations()
//     {
//         while (_isPollingAnnotations)
//         {
//             yield return new WaitForSeconds(_annotationPollInterval);
//             yield return StartCoroutine(GetLatestAnnotationCoroutine());
//         }
//     }

//     private IEnumerator GetLatestAnnotationCoroutine()
//     {
//         string url = _serverUri + "/get-latest-annotation";

//         using (UnityWebRequest request = UnityWebRequest.Get(url))
//         {
//             request.certificateHandler = new AcceptAnyCertificate();
//             request.disposeCertificateHandlerOnDispose = true;

//             yield return request.SendWebRequest();

//             if (request.result == UnityWebRequest.Result.Success)
//             {
//                 try
//                 {
//                     string responseText = request.downloadHandler.text;
//                     Debug.Log($"📥 Annotation response: {responseText}");

//                     AnnotationResponse response = JsonUtility.FromJson<AnnotationResponse>(responseText);

//                     if (response.status == "success" && response.annotation != null)
//                     {
//                         Debug.Log($"🎯 New annotation received: ID {response.annotation.id}");
//                         Debug.Log($"📊 Center: ({response.annotation.center[0]}, {response.annotation.center[1]})");
//                         Debug.Log($"📐 Pose matrix elements: {response.annotation.pose_matrix?.Length ?? 0}");

//                         OnNewAnnotationReceived?.Invoke(response.annotation);
//                     }
//                     else if (response.status == "no_annotations")
//                     {
//                         // No new annotations - this is normal, don't spam logs
//                     }
//                     else if (response.status == "already_processed")
//                     {
//                         // Already processed - this is normal
//                     }
//                     else
//                     {
//                         Debug.LogWarning($"⚠️ Unexpected annotation response status: {response.status}");
//                     }
//                 }
//                 catch (System.Exception e)
//                 {
//                     Debug.LogError($"❌ Error parsing annotation response: {e.Message}");
//                     OnAnnotationError?.Invoke($"Failed to parse annotation: {e.Message}");
//                 }
//             }
//             else
//             {
//                 Debug.LogWarning($"⚠️ Failed to get annotations: {request.error}");
//                 OnAnnotationError?.Invoke($"Network error: {request.error}");
//             }
//         }
//     }

//     private IEnumerator MarkAnnotationsProcessedCoroutine(int[] annotationIds)
//     {
//         string url = _serverUri + "/mark-annotations-processed";

//         ProcessedAnnotations payload = new ProcessedAnnotations
//         {
//             processed_ids = annotationIds
//         };

//         string jsonPayload = JsonUtility.ToJson(payload);
//         byte[] postData = Encoding.UTF8.GetBytes(jsonPayload);

//         using (UnityWebRequest request = new UnityWebRequest(url, "POST"))
//         {
//             request.uploadHandler = new UploadHandlerRaw(postData);
//             request.downloadHandler = new DownloadHandlerBuffer();
//             request.SetRequestHeader("Content-Type", "application/json");

//             request.certificateHandler = new AcceptAnyCertificate();
//             request.disposeCertificateHandlerOnDispose = true;

//             yield return request.SendWebRequest();

//             if (request.result == UnityWebRequest.Result.Success)
//             {
//                 Debug.Log($"✅ Marked {annotationIds.Length} annotations as processed");
//             }
//             else
//             {
//                 Debug.LogWarning($"⚠️ Failed to mark annotations as processed: {request.error}");
//             }
//         }
//     }

//     public void Disconnect()
//     {
//         StopAnnotationPolling();
//         DisconnectAudio();
//         _webRequestManager.HttpPost(_serverUri + "/logout/" + _localId, string.Empty);
//     }

//     private void OnDestroy()
//     {
//         DisconnectAudio();
//     }
// }