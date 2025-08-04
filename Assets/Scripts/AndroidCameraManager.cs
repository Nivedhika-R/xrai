using System.Collections;
using UnityEngine;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;
using UnityEngine.Experimental.Rendering;

/// <summary>
/// Manages obtaining a render texture from AR camera (ARFoundation) in Unity.
/// This includes initializing the camera and converting camera frames to a RenderTexture.
/// </summary>
public class AndroidCameraManager : MonoBehaviour, ICameraDeviceManager
{
    [SerializeField, Tooltip("The renderer to show the camera capture on RGB format")]
    private Renderer _screenRendererRGB = null;

    [SerializeField, Tooltip("AR Camera Manager for accessing camera frames")]
    private ARCameraManager arCameraManager;

    private bool _isCameraConfiguredAndReady = false;
    private RenderTexture _readTexture;
    private Texture2D _cameraTexture;

    private const int WIDTH = 1280;
    private const int HEIGHT = 720;

    private bool _isProcessingImage = false;

    private Matrix4x4 _cameraToWorldMatrix = Matrix4x4.identity;
    private Matrix4x4 _intrinsicsMatrix = Matrix4x4.identity;

    public RenderTexture CameraTexture => _readTexture;
    public Matrix4x4 CameraToWorldMatrix => _cameraToWorldMatrix;
    public Matrix4x4 Instrinsics => _intrinsicsMatrix;
    public Matrix4x4 Distortion => Matrix4x4.identity;

    public bool IsConfiguredAndReady => _isCameraConfiguredAndReady;

    public void StartMedia()
    {
        if (arCameraManager == null)
        {
            Debug.LogError("ARCameraManager is not assigned.");
            return;
        }

        arCameraManager.frameReceived += OnCameraFrameReceived;

        if (_screenRendererRGB != null && _screenRendererRGB.gameObject.activeInHierarchy)
        {
            _screenRendererRGB.material.mainTexture = _readTexture;
        }

        _isCameraConfiguredAndReady = true;
        Debug.Log("Camera started and ready to receive frames.");
    }

    public void StopMedia()
    {
        StopCamera();
    }

    private void StopCamera()
    {
        if (_isCameraConfiguredAndReady && arCameraManager != null)
        {
            arCameraManager.frameReceived -= OnCameraFrameReceived;
            _isCameraConfiguredAndReady = false;
        }
    }

    private void OnCameraFrameReceived(ARCameraFrameEventArgs args)
    {
        _cameraToWorldMatrix = arCameraManager.transform.localToWorldMatrix;

        if (arCameraManager.TryGetIntrinsics(out XRCameraIntrinsics intrinsics))
        {
            _intrinsicsMatrix = new Matrix4x4(
                new Vector4(intrinsics.focalLength.x, 0, 0, 0),
                new Vector4(0, intrinsics.focalLength.y, 0, 0),
                new Vector4(intrinsics.principalPoint.x, intrinsics.principalPoint.y, 1, 0),
                new Vector4(0, 0, 0, 1)
            );
        }

        if (_isProcessingImage)
        {
            // Debug.LogWarning("Already processing an image, skipping this frame.");
            return;
        }

        if (!arCameraManager.TryAcquireLatestCpuImage(out XRCpuImage image))
        {
            Debug.LogError("Failed to acquire latest CPU image.");
            return;
        }

        _isProcessingImage = true;
        StartCoroutine(ProcessCameraImage(image));
    }

    private IEnumerator ProcessCameraImage(XRCpuImage image)
    {
        var conversionParams = new XRCpuImage.ConversionParams
        {
            inputRect = new RectInt(0, 0, image.width, image.height),
            outputDimensions = new Vector2Int(image.width, image.height),
            outputFormat = TextureFormat.RGBA32,
            transformation = XRCpuImage.Transformation.MirrorY
        };

        // Recreate _cameraTexture if needed
        if (_cameraTexture == null || _cameraTexture.width != image.width || _cameraTexture.height != image.height)
        {
            _cameraTexture = new Texture2D(image.width, image.height, TextureFormat.RGBA32, false);
        }

        var rawTextureData = _cameraTexture.GetRawTextureData<byte>();
        image.Convert(conversionParams, rawTextureData);
        image.Dispose();

        _cameraTexture.Apply();

        // Recreate _readTexture if needed
        if (_readTexture == null || _readTexture.width != image.width || _readTexture.height != image.height)
        {
            _readTexture?.Release();
            _readTexture = new RenderTexture(image.width, image.height, 0, GraphicsFormat.R8G8B8A8_UNorm);
            if (_screenRendererRGB != null && _screenRendererRGB.gameObject.activeInHierarchy)
            {
                _screenRendererRGB.material.mainTexture = _readTexture;
            }
        }

        Graphics.Blit(_cameraTexture, _readTexture);

        yield return null;

        _isProcessingImage = false;
    }

    public bool IsMediaReady()
    {
        return _isCameraConfiguredAndReady;
    }
}
