using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public interface ICameraDeviceManager
{
    public RenderTexture CameraTexture { get; }
    public Matrix4x4 CameraToWorldMatrix { get; }
    public Matrix4x4 ProjectionMatrix { get; }

    public bool IsConfiguredAndReady { get; }

    public void StartMedia();
    public void StopMedia();


}
