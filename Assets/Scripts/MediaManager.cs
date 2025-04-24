using System;
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

    public IEnumerator ReadRenderTextureimageBytesAsync(RenderTexture renderTexture, Action<byte[]> callback, bool png = false)
    {
        // Backup the currently active RenderTexture
        RenderTexture currentRT = RenderTexture.active;
        RenderTexture.active = renderTexture;

        // Create the Texture2D
        Texture2D tex = new(renderTexture.width, renderTexture.height, TextureFormat.RGB24, false);

        // Read pixels from RenderTexture (still runs on main thread)
        tex.ReadPixels(new Rect(0, 0, renderTexture.width, renderTexture.height), 0, 0);
        tex.Apply();

        // Wait for a frame to avoid blocking the current frame
        yield return null;

        // Encode texture to PNG or JPG
        byte[] imageBytes;
        if (png)
            imageBytes = tex.EncodeToPNG();
        else
            imageBytes = tex.EncodeToJPG(85);

        // Restore the previous RenderTexture
        RenderTexture.active = currentRT;
        UnityEngine.Object.Destroy(tex);

        // Return result via callback
        callback?.Invoke(imageBytes);
    }

    public void GetImage(Action<Matrix4x4,Matrix4x4,Matrix4x4,byte[]> callback, bool png = false)
    {
        Matrix4x4 cameraToWorldMatrix = _targetCameraDeviceManager.CameraToWorldMatrix;
        Matrix4x4 instrinsics = _targetCameraDeviceManager.Instrinsics;
        Matrix4x4 distortion = _targetCameraDeviceManager.Distortion;
        StartCoroutine(
            ReadRenderTextureimageBytesAsync(_targetCameraDeviceManager.CameraTexture, (imageBytes) =>
                {
                    callback?.Invoke(cameraToWorldMatrix, instrinsics, distortion, imageBytes);
                },
            png)
        );
    }
}
