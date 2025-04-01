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

    // Stores: Key = log tag, Value = (Message, Timestamp, Type)
    private Dictionary<string, (string Value, string Timestamp, string Type)> debugLogs =
        new Dictionary<string, (string, string, string)>();

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

        string timestamp = DateTime.Now.ToString("HH:mm:ss");
        string typeLabel = GetLogTypeLetter(type);

        if (debugLogs.ContainsKey(key))
            debugLogs[key] = (value, timestamp, typeLabel);
        else
            debugLogs.Add(key, (value, timestamp, typeLabel));

        UpdateDebugDisplay();
    }

    private void UpdateDebugDisplay()
    {
        ClearLinesIfNeeded();

        string output = "";
        foreach (var log in debugLogs)
        {
            string time = log.Value.Timestamp;
            string type = log.Value.Type;
            string value = log.Value.Value;

            string line = string.IsNullOrEmpty(value)
                ? $"[{time}] [{type}] {log.Key}"
                : $"[{time}] [{type}] {log.Key}: {value}";

            output += line + "\n";
        }

        debugAreaText.text = output;
    }

    public void Log(string message)
    {
        LogInternal(message, "I");
    }

    public void LogWarning(string message)
    {
        LogInternal(message, "W");
    }

    public void LogError(string message)
    {
        LogInternal(message, "E");
    }

    private void LogInternal(string message, string type)
    {
        if (!enableDebug) return;

        ClearLinesIfNeeded();

        string timestamp = DateTime.Now.ToString("HH:mm:ss");
        debugAreaText.text += $"[{timestamp}] [{type}] {message}\n";
    }

    private void ClearLinesIfNeeded()
    {
        string[] lines = debugAreaText.text.Split('\n');
        if (lines.Length >= maxLines)
        {
            // Keep only error logs
            var errorLogs = debugLogs
                .Where(kvp => kvp.Value.Type == "E")
                .ToDictionary(kvp => kvp.Key, kvp => kvp.Value);

            debugLogs.Clear();
            foreach (var err in errorLogs)
                debugLogs.Add(err.Key, err.Value);

            debugAreaText.text = "";
        }
    }

    private string GetLogTypeLetter(LogType type)
    {
        switch (type)
        {
            case LogType.Error:
            case LogType.Exception:
                return "E";
            case LogType.Warning:
                return "W";
            default:
                return "I";
        }
    }
}
