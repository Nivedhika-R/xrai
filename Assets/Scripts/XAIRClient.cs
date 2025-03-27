using MagicLeap;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Unity.WebRTC;
using UnityEngine;

public class XAIRClient : Singleton<XAIRClient>
{
    [SerializeField] private string ipAddress = "http://0.0.0.0:8080";

    public Action<bool, string> OnWebRTCConnectionChanged;

    void Start() {
        Debug.Log("XAIRClient starting at " + ipAddress);

        WebRTCController.Instance.OnWebRTCConnectionStateChange += OnWebRTCConnectionStateChanged;

        // connect after 1 second
        StartCoroutine(Connect());
    }

    private IEnumerator Connect()
    {
        yield return new WaitForSeconds(1);
        OnWebRTCConnectionChanged?.Invoke(true, ipAddress);
    }

    private void OnWebRTCConnectionStateChanged(WebRTCController.WebRTCConnectionState connectionState)
    {
        Debug.Log("WebRTC connection state changed to " + connectionState);
    }
};
