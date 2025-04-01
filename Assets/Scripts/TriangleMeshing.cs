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
    private ARMeshManager _meshingManager;

    private MagicLeapMeshingFeature meshingFeature;

    // Search for the Mesh Manager and assign it automatically if it was not set in the inspector
    private void OnValidate()
    {
        if (_meshingManager == null)
        {
            _meshingManager = FindObjectOfType<ARMeshManager>();
        }
    }

    // Coroutine to initialize meshing after ensuring the necessary subsystems and permissions are available
    IEnumerator Start()
    {
        // Check if the ARMeshManager component is assigned, if not, try to find one in the scene
        if (_meshingManager == null)
        {
            Debug.LogError("No ARMeshManager component found. Disabling script.");
            enabled = false;
            yield break;
        }

        // Disable the mesh manager until permissions are granted
        _meshingManager.enabled = false;

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

    public bool RayCastToMesh(Ray ray, out Vector3 hitPoint)
    {
        hitPoint = Vector3.zero;

        if (_meshingManager == null || !_meshingManager.enabled)
        {
            Debug.LogError("Mesh Manager is not enabled.");
            return false;
        }

        foreach (var meshFilter in _meshingManager.meshes)
        {
            if (meshFilter == null) continue;

            var meshCollider = meshFilter.GetComponent<MeshCollider>();
            if (meshCollider == null)
            {
                Debug.LogWarning("MeshCollider missing on mesh.");
                continue;
            }

            if (!meshCollider.enabled)
            {
                Debug.LogWarning("MeshCollider is disabled.");
                continue;
            }

            // Ensure mesh is assigned
            if (meshCollider.sharedMesh == null)
                meshCollider.sharedMesh = meshFilter.sharedMesh;

            if (meshCollider.Raycast(ray, out RaycastHit hit, Mathf.Infinity))
            {
                hitPoint = hit.point;
                return true;
            }
        }

        return false;
    }

    private void OnPermissionGranted(string permission)
    {
        _meshingManager.enabled = true;
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
