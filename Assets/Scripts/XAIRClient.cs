using MagicLeap;
using SimpleJson;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using TMPro;
using UnityEngine;

public class XAIRClient : Singleton<XAIRClient>
{
    [SerializeField] private ServerCommunication _serverCommunication;
    [SerializeField] private MediaManager _mediaManager;

    [SerializeField] private string ipAddress = "0.0.0.0";
    [SerializeField] private string port = "8000";
    [SerializeField] private bool noSSL = false;

    [SerializeField] private TextMeshProUGUI llmAreaText = null;

    private string _serverUri;

    public Action<bool, string> OnWebRTCConnectionChanged;

    private float _sendImagetimer = 0.0f;
    private float _sendImageFreqHz = 5.0f;

    void Start() {
        _serverUri = noSSL ? "http://" + ipAddress + ":" + port : "https://" + ipAddress + ":" + port;
        Debug.Log("XAIRClient starting at " + _serverUri);

        WebRTCController.Instance.OnWebRTCConnectionStateChange += OnWebRTCConnectionStateChanged;
        WebRTCController.Instance.OnDataChannelMessageReceived += OnDataChannelMessageReceived;

        // connect after 1 second
        StartCoroutine(Connect());
    }

    // every 1 second send image to server
    void Update()
    {
        if (_serverCommunication != null)
        {
            _sendImagetimer += Time.deltaTime;
            if (_sendImagetimer >= 1.0f / _sendImageFreqHz)
            {
                _sendImagetimer = 0.0f;

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

        OnWebRTCConnectionChanged?.Invoke(true, _serverUri);
    }

    private void OnWebRTCConnectionStateChanged(WebRTCController.WebRTCConnectionState connectionState)
    {
        Debug.Log("WebRTC connection state changed to " + connectionState);
    }

    private void OnDataChannelMessageReceived(string message)
    {
        Debug.Log("Data channel message received: " + message);
        SimpleJson.SimpleJson.TryDeserializeObject(message, out object obj);
        JsonObject jsonObj = (JsonObject)obj;
        if (jsonObj.ContainsKey("type"))
        {
            string msgType = jsonObj["type"].ToString();
            if (msgType == "greeting")
            {
                string greeting = jsonObj["content"].ToString();
                UpdateLLMAreaText(greeting);
            }
            else
            if (msgType == "llm_reply")
            {
                string llm_reply = jsonObj["content"].ToString();
                string client_id = jsonObj["client_id"].ToString();
                string timestamp = jsonObj["timestamp"].ToString();
                string cameraToWorldMatrix = jsonObj["cameraToWorldMatrix"].ToString();
                string projectionMatrix = jsonObj["projectionMatrix"].ToString();

                UpdateLLMAreaText(llm_reply);
            }
        }
    }

    private void UpdateLLMAreaText(string text)
    {
        if (llmAreaText != null)
        {
            llmAreaText.text = text;
        }
    }
};
