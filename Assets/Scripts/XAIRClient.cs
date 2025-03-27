using MagicLeap;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Unity.WebRTC;
using UnityEngine;

public class XAIRClient : Singleton<XAIRClient>
{
    [SerializeField] private ServerCommunication _serverCommunication;
    [SerializeField] private MediaManager _mediaManager;

    [SerializeField] private string ipAddress = "0.0.0.0";
    [SerializeField] private string port = "8000";
    [SerializeField] private bool noSSL = false;

    private string serverAdress;

    public Action<bool, string> OnWebRTCConnectionChanged;


    private float _timer = 0.0f;
    private float _sendImageFreqHz = 5.0f;

    void Start() {
        serverAdress = noSSL ? "http://" + ipAddress + ":" + port : "https://" + ipAddress + ":" + port;
        Debug.Log("XAIRClient starting at " + serverAdress);

        WebRTCController.Instance.OnWebRTCConnectionStateChange += OnWebRTCConnectionStateChanged;

        // connect after 1 second
        StartCoroutine(Connect());
    }

    // every 1 second send image to server
    void Update()
    {
        if (_serverCommunication != null)
        {
            _timer += Time.deltaTime;
            if (_timer >= 1.0f / _sendImageFreqHz)
            {
                _timer = 0.0f;

                Matrix4x4 cameraToWorldMatrix, projectionMatrix;
                byte[] imageBytes = _mediaManager.GetImage(out cameraToWorldMatrix, out projectionMatrix);
                if (imageBytes != null)
                {
                    _serverCommunication.SendImage(imageBytes, cameraToWorldMatrix, projectionMatrix);
                }
            }
        }
    }

    private IEnumerator Connect()
    {
        // wait until media is ready
        while (!MediaManager.Instance.IsMediaReady())
        {
            yield return new WaitForSeconds(1);
        }

        OnWebRTCConnectionChanged?.Invoke(true, serverAdress);
    }

    private void OnWebRTCConnectionStateChanged(WebRTCController.WebRTCConnectionState connectionState)
    {
        Debug.Log("WebRTC connection state changed to " + connectionState);
    }
};
