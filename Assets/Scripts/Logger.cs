using System;
using System.Collections.Generic;
using System.Linq;
using TMPro;
using UnityEngine;
using MagicLeap;

public class Logger : Singleton<Logger>
{
    [SerializeField] private TextMeshProUGUI debugAreaText = null;
    [SerializeField] private bool enableDebug = false;
    [SerializeField] private int maxLines = 20;

    private Dictionary<string, string> debugLogs = new Dictionary<string, string>();
    private Dictionary<string, string> debugTimestamps = new Dictionary<string, string>();

    private void Start()
    {
        if (debugAreaText == null)
            debugAreaText = GetComponent<TextMeshProUGUI>();
    }

    private void OnEnable()
    {
        Application.logMessageReceived += HandleLog;
        debugAreaText.enabled = enableDebug;
        enabled = enableDebug;

        if (enabled)
        {
            Log("Logger enabled!");
        }
    }

    private void OnDisable()
    {
        Application.logMessageReceived -= HandleLog;
    }

    private void HandleLog(string logString, string stackTrace, LogType type)
    {
        if (!enableDebug) return;

        string[] splitString = logString.Split(':');
        string key = splitString[0];
        string value = splitString.Length > 1 ? splitString[1].Trim() : "";

        if (debugLogs.ContainsKey(key))
            debugLogs[key] = value;
        else
            debugLogs.Add(key, value);

        debugTimestamps[key] = DateTime.Now.ToString("HH:mm:ss.fff");

        UpdateDebugDisplay();
    }

    private void UpdateDebugDisplay()
    {
        ClearLinesIfNeeded();

        string output = "";
        foreach (var log in debugLogs)
        {
            string time = debugTimestamps.ContainsKey(log.Key) ? debugTimestamps[log.Key] : "--:--:--";
            if (string.IsNullOrEmpty(log.Value))
                output += $"{time} {log.Key}\n";
            else
                output += $"{time} {log.Key}: {log.Value}\n";
        }

        debugAreaText.text = output;
    }

    public void Log(string message)
    {
        LogInternal(message, "LOG");
    }

    public void LogWarning(string message)
    {
        LogInternal(message, "WARN");
    }

    public void LogError(string message)
    {
        LogInternal(message, "ERROR");
    }

    private void LogInternal(string message, string type)
    {
        if (!enableDebug) return;

        ClearLinesIfNeeded();

        string timestamp = DateTime.Now.ToString("HH:mm:ss.fff");
        debugAreaText.text += $"{type}: {timestamp} {message}\n";
    }

    private void ClearLinesIfNeeded()
    {
        if (debugAreaText.text.Split('\n').Length >= maxLines)
        {
            debugAreaText.text = string.Empty;
            debugLogs.Clear();
            debugTimestamps.Clear();
        }
    }
}
