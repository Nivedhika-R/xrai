using System.Collections;
using UnityEngine;
using MagicLeap;
using UnityEngine.UI;

/// <summary>
/// Manages media components such as camera and microphone for both device and editor platforms.
/// </summary>
public class MediaManager : Singleton<MediaManager>
{
    [Header("Remote Media")]
    [SerializeField] private RawImage _remoteVideoRenderer;
    [SerializeField] public AudioSource  _receiveAudio;

    [Header("Local Media")]

    [SerializeReference]
    private MagicLeapCameraManager _magicLeapCameraDeviceManager;
    [SerializeReference]
    private WebCamManager _webCamDeviceManager;
    [SerializeField]
    private MicrophoneManager _microphoneManager;

    [Header("Permissions")]

    [SerializeField]
    private PermissionsManager _permissionsManager;

    [Header("Magic Leap Settings")]

    [SerializeField]
    [Tooltip("Will use the MLCamera APIs instead of the WebCamera Texture component.")]
    private bool _useMLCamera = true;

    private ICameraDeviceManager _targetCameraDeviceManager;

    public RawImage RemoteVideoRenderer => _remoteVideoRenderer;

    public RenderTexture CameraTexture => _targetCameraDeviceManager.CameraTexture;

    public AudioSource SourceAudio => _microphoneManager.SourceAudio;

    public AudioSource ReceiveAudio => _receiveAudio;

    private IEnumerator Start()
    {

        if (_useMLCamera && Application.platform == RuntimePlatform.Android
            && SystemInfo.deviceModel == "Magic Leap Magic Leap 2")
        {
            _targetCameraDeviceManager = _magicLeapCameraDeviceManager;
        }
        else
        {
            _targetCameraDeviceManager = _webCamDeviceManager;
        }

        _permissionsManager.RequestPermission();
        yield return new WaitUntil(() => _permissionsManager.PermissionsGranted);

        StartMedia();
    }

    private void StartMedia()
    {
        _targetCameraDeviceManager.StartMedia();
        _microphoneManager.SetupAudio();
    }

    private void OnDisable()
    {
        StopMedia();
    }

    /// <summary>
    /// Stop microphone and camera on Magic Leap 2
    /// TODO: Implement UI to actually call this function
    /// </summary>
    private void StopMedia()
    {
        _targetCameraDeviceManager.StopMedia();
        _microphoneManager.StopMicrophone();

    }

    /// <summary>
    /// Returns true if local microphone and camera are ready
    /// </summary>
    public bool IsMediaReady()
    {
        bool res = _targetCameraDeviceManager != null
                && _targetCameraDeviceManager.IsConfiguredAndReady
                && _microphoneManager.IsConfiguredAndReady;
        if (res) {
            Debug.Log($" Camera Device Ready = {_targetCameraDeviceManager.IsConfiguredAndReady} && Microphone Ready = {_microphoneManager.IsConfiguredAndReady} ");
        }

        return res;
    }

    public byte[] ReadRenderTextureBytes(RenderTexture renderTexture, bool png = false)
    {
        // Set active RenderTexture
        RenderTexture currentRT = RenderTexture.active;
        RenderTexture.active = renderTexture;

        // Create a new Texture2D with same dimensions
        Texture2D tex = new Texture2D(renderTexture.width, renderTexture.height, TextureFormat.RGB24, false);

        // Read pixels from the RenderTexture
        tex.ReadPixels(new Rect(0, 0, renderTexture.width, renderTexture.height), 0, 0);
        tex.Apply();

        // Encode texture to PNG (you can also use EncodeToJPG or GetRawTextureData)
        byte[] bytes;
        if (png)
            bytes = tex.EncodeToPNG();
        else
            bytes = tex.EncodeToJPG(85);

        // Clean up
        RenderTexture.active = currentRT;
        Destroy(tex); // if you're in a coroutine or function outside of OnDestroy

        return bytes;
    }

    public byte[] GetImage(out Matrix4x4 cameraToWorldMatrix, out Matrix4x4 instrinsics, out Matrix4x4 distortion, bool png = false)
    {
        byte[] imageBytes = ReadRenderTextureBytes(_targetCameraDeviceManager.CameraTexture, png);
        cameraToWorldMatrix = _targetCameraDeviceManager.CameraToWorldMatrix;
        instrinsics = _targetCameraDeviceManager.Instrinsics;
        distortion = _targetCameraDeviceManager.Distortion;
        return imageBytes;
    }
}
