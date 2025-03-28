using System.Collections;
using UnityEngine;
using UnityEngine.XR;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.MagicLeap;
using UnityEngine.XR.Management;
using UnityEngine.XR.OpenXR;
using MagicLeap.Android;
using MagicLeap.OpenXR.Features.Meshing;

public class TriangleMeshing : MonoBehaviour
{
    [SerializeField]
    private ARMeshManager _meshManager;

    private MagicLeapMeshingFeature meshingFeature;

    // Search for the Mesh Manager and assign it automatically if it was not set in the inspector
    private void OnValidate()
    {
        if (_meshManager == null)
        {
            _meshManager = FindObjectOfType<ARMeshManager>();
        }
    }

    // Coroutine to initialize meshing after ensuring the necessary subsystems and permissions are available
    IEnumerator Start()
    {
        // Check if the ARMeshManager component is assigned, if not, try to find one in the scene
        if (_meshManager == null)
        {
            Debug.LogError("No ARMeshManager component found. Disabling script.");
            enabled = false;
            yield break;
        }

        // Disable the mesh manager until permissions are granted
        _meshManager.enabled = false;

        yield return new WaitUntil(IsMeshingSubsystemLoaded);

        // Magic Leap specific Meshing features can be accessed using this class
        meshingFeature = OpenXRSettings.Instance.GetFeature<MagicLeapMeshingFeature>();
        if (!meshingFeature.enabled)
        {
            Debug.LogError("MagicLeapMeshingFeature was not enabled. Disabling script");
            enabled = false;
            yield break;
        }

        Permissions.RequestPermission(Permissions.SpatialMapping, OnPermissionGranted, OnPermissionDenied, OnPermissionDenied);
    }

    private void OnPermissionGranted(string permission)
    {
        _meshManager.enabled = true;
    }

    private void OnPermissionDenied(string permission)
    {
        Debug.LogError($"Permission {Permissions.SpatialMapping} denied. Disabling script.");
        enabled = false;
    }

    private bool IsMeshingSubsystemLoaded()
    {
        if (XRGeneralSettings.Instance == null || XRGeneralSettings.Instance.Manager == null) return false;
        var activeLoader = XRGeneralSettings.Instance.Manager.activeLoader;
        return activeLoader != null && activeLoader.GetLoadedSubsystem<XRMeshSubsystem>() != null;
    }
}
