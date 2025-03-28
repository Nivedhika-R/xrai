using System.Collections.Generic;
using MagicLeap.Android;
using UnityEngine;
using UnityEngine.Android;

public class PermissionsManager : MonoBehaviour
{
    [SerializeField]
    private List<string> _requiredPermissions = new List<string> { Permission.Microphone, Permission.Camera, Permissions.SpatialMapping };

    public bool PermissionsGranted => Permissions.CheckPermission(Permission.Microphone)
                                      && Permissions.CheckPermission(Permission.Camera);

    private void OnValidate()
    {
        // Ensure that the required permissions list contains Microphone and Camera permissions
        var required = new List<string> { Permission.Microphone, Permission.Camera };
        foreach (var permission in required)
        {
            if (!_requiredPermissions.Contains(permission))
            {
                Debug.LogError($"Permission {permission} is required. Adding it to the list.");
                _requiredPermissions.Add(permission);
            }
        }
    }

    void Awake()
    {
        RequestPermission();
    }

    public void RequestPermission()
    {
        if(!PermissionsGranted)
            Permissions.RequestPermissions(_requiredPermissions.ToArray(), OnPermissionGranted, OnPermissionDenied, OnPermissionDenied);
    }

    void OnPermissionGranted(string permission)
    {
        Debug.Log($"{permission} granted.");

    }
    void OnPermissionDenied(string permission)
    {
        Debug.LogError($"{permission} denied, example won't function.");
    }
}
