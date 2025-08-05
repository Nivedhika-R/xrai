using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR.ARFoundation;

public class ARPlaceCube : MonoBehaviour
{
    [Header("AR Components")]
    [SerializeField] private ARRaycastManager raycastManager;

    [Header("Server Communication")]
    [SerializeField] private ServerCommunication serverCommunication;

    [Header("Annotation Settings")]
    [SerializeField] private GameObject annotationPrefab; // Different prefab for annotations
    [SerializeField] private float annotationHitTestDistance = 3.0f;
    [SerializeField] private bool enableAnnotationSystem = true;

    [Header("Debug")]
    [SerializeField] private bool debugAnnotations = true;

    // Annotation system variables
    private List<GameObject> placedAnnotations = new List<GameObject>();
    private List<int> processedAnnotationIds = new List<int>();

    void Start()
    {
        // Initialize ONLY annotation system
        if (enableAnnotationSystem && serverCommunication != null)
        {
            InitializeAnnotationSystem();
        }
        else if (enableAnnotationSystem && serverCommunication == null)
        {
            Debug.LogError("❌ ServerCommunication reference is null! Please assign it in the inspector.");
            enableAnnotationSystem = false;
        }

        if (annotationPrefab == null)
        {
            Debug.LogWarning("⚠️ No annotation prefab assigned! Please create and assign a prefab.");
        }
    }

    void OnDestroy()
    {
        // Clean up annotation system
        if (enableAnnotationSystem && serverCommunication != null)
        {
            serverCommunication.OnNewAnnotationReceived -= HandleNewAnnotation;
            serverCommunication.OnAnnotationError -= HandleAnnotationError;
        }
    }

    private void InitializeAnnotationSystem()
    {
        // Subscribe to annotation events
        serverCommunication.OnNewAnnotationReceived += HandleNewAnnotation;
        serverCommunication.OnAnnotationError += HandleAnnotationError;

        // Start polling for annotations every 2 seconds
        serverCommunication.StartAnnotationPolling(2.0f);

        Debug.Log("🎯 ARPlaceCube: Annotation system initialized and polling started");
    }

    private void HandleNewAnnotation(ServerCommunication.AnnotationData annotation)
    {
        if (debugAnnotations)
        {
            Debug.Log("═══════════════════════════════════════");
            Debug.Log($"🎯 NEW ANNOTATION RECEIVED: ID {annotation.id}");
            Debug.Log($"📊 2D Center: ({annotation.center[0]}, {annotation.center[1]})");
            Debug.Log($"📦 Bounding box: {annotation.bounding_box.width}x{annotation.bounding_box.height}");
            Debug.Log($"📐 Pose matrix elements: {annotation.pose_matrix?.Length ?? 0}");
            Debug.Log($"💡 Will place at ORIGINAL camera pose location");
            Debug.Log("═══════════════════════════════════════");
        }

        // Convert using ORIGINAL camera pose (not current camera position)
        Vector3 worldPosition = ConvertAnnotationTo3DUsingOriginalPose(annotation);

        if (worldPosition != Vector3.zero)
        {
            PlaceAnnotationInWorld(worldPosition, annotation);
            processedAnnotationIds.Add(annotation.id);
            serverCommunication.MarkAnnotationsAsProcessed(new int[] { annotation.id });

            if (debugAnnotations)
            {
                Debug.Log($"✅ Successfully placed annotation {annotation.id}");
                Debug.Log($"📍 World position: {worldPosition}");
                Debug.Log($"🎯 This represents where user annotated in the original view");
            }
        }
        else
        {
            Debug.LogWarning($"⚠️ Could not place annotation {annotation.id} - conversion failed");
        }
    }

    private void HandleAnnotationError(string error)
    {
        Debug.LogWarning($"⚠️ Annotation system error: {error}");
    }

    private Vector3 ConvertAnnotationTo3DUsingOriginalPose(ServerCommunication.AnnotationData annotation)
    {
        try
        {
            float screenX = annotation.center[0];
            float screenY = annotation.center[1];

            if (debugAnnotations)
            {
                Debug.Log($"🔄 Converting screen coords ({screenX}, {screenY})");
                Debug.Log($"🎯 Using ORIGINAL camera pose, not current camera position");
            }

            // Method 1: Try AR Raycast (works if camera is in similar position)
            Vector3 worldPos = TryARRaycastForAnnotation(screenX, screenY);
            if (worldPos != Vector3.zero)
            {
                if (debugAnnotations) Debug.Log($"✅ AR Raycast successful: {worldPos}");
                return worldPos;
            }

            // Method 2: Use ORIGINAL pose matrix (most accurate)
            if (annotation.pose_matrix != null && annotation.pose_matrix.Length >= 16)
            {
                worldPos = CalculateWorldPositionFromOriginalPose(screenX, screenY, annotation.pose_matrix);
                if (worldPos != Vector3.zero)
                {
                    if (debugAnnotations) Debug.Log($"✅ Pose matrix calculation: {worldPos}");
                    return worldPos;
                }
            }

            // Method 3: Simple fallback
            worldPos = SimpleDepthEstimation(screenX, screenY);
            if (debugAnnotations) Debug.Log($"⚠️ Using fallback estimation: {worldPos}");
            return worldPos;

        }
        catch (System.Exception e)
        {
            Debug.LogError($"❌ Error converting annotation to 3D: {e.Message}");
            return Vector3.zero;
        }
    }

    private Vector3 TryARRaycastForAnnotation(float screenX, float screenY)
    {
        if (raycastManager == null) return Vector3.zero;

        Vector2 screenPosition = new Vector2(screenX, screenY);
        var rayHits = new List<ARRaycastHit>();

        if (raycastManager.Raycast(screenPosition, rayHits, UnityEngine.XR.ARSubsystems.TrackableType.AllTypes))
        {
            return rayHits[0].pose.position;
        }

        return Vector3.zero;
    }

    private Vector3 CalculateWorldPositionFromOriginalPose(float screenX, float screenY, float[] poseMatrix)
    {
        try
        {
            Camera arCamera = Camera.main;
            if (arCamera == null)
            {
                Debug.LogWarning("⚠️ Camera.main is null, trying to find AR Camera");
                arCamera = FindObjectOfType<Camera>();
                if (arCamera == null) return Vector3.zero;
            }

            // Convert pose matrix to Unity Matrix4x4
            Matrix4x4 originalCameraToWorld = ArrayToMatrix4x4(poseMatrix);

            if (debugAnnotations)
            {
                Vector3 originalPos = new Vector3(originalCameraToWorld.m03, originalCameraToWorld.m13, originalCameraToWorld.m23);
                Debug.Log($"📐 Original camera position: {originalPos}");
                Debug.Log($"📐 Current camera position: {arCamera.transform.position}");
            }

            // Get screen dimensions
            float screenWidth = Screen.width;
            float screenHeight = Screen.height;

            // Convert screen coordinates to normalized coordinates (-1 to 1)
            float normalizedX = (screenX / screenWidth) * 2.0f - 1.0f;
            float normalizedY = (screenY / screenHeight) * 2.0f - 1.0f;

            // Extract camera position and rotation from original pose matrix
            Vector3 originalCameraPos = new Vector3(originalCameraToWorld.m03, originalCameraToWorld.m13, originalCameraToWorld.m23);
            Vector3 originalCameraForward = new Vector3(-originalCameraToWorld.m02, -originalCameraToWorld.m12, -originalCameraToWorld.m22);
            Vector3 originalCameraRight = new Vector3(originalCameraToWorld.m00, originalCameraToWorld.m10, originalCameraToWorld.m20);
            Vector3 originalCameraUp = new Vector3(originalCameraToWorld.m01, originalCameraToWorld.m11, originalCameraToWorld.m21);

            // Calculate ray direction based on screen coordinates and original camera orientation
            Vector3 rayDirection = originalCameraForward +
                                 normalizedX * originalCameraRight * 0.5f +
                                 normalizedY * originalCameraUp * 0.5f;
            rayDirection.Normalize();

            // Calculate world position using estimated depth
            Vector3 worldPosition = originalCameraPos + rayDirection * annotationHitTestDistance;

            return worldPosition;
        }
        catch (System.Exception e)
        {
            Debug.LogError($"❌ Pose calculation error: {e.Message}");
            return Vector3.zero;
        }
    }

    private Vector3 SimpleDepthEstimation(float screenX, float screenY)
    {
        Camera arCamera = Camera.main;
        if (arCamera == null)
        {
            arCamera = FindObjectOfType<Camera>();
            if (arCamera == null) return Vector3.zero;
        }

        return arCamera.ScreenToWorldPoint(new Vector3(screenX, screenY, annotationHitTestDistance));
    }

    private Matrix4x4 ArrayToMatrix4x4(float[] array)
    {
        if (array.Length < 16) return Matrix4x4.identity;

        Matrix4x4 matrix = new Matrix4x4();
        for (int i = 0; i < 16; i++)
        {
            int row = i / 4;
            int col = i % 4;
            matrix[row, col] = array[i];
        }
        return matrix;
    }

    private void PlaceAnnotationInWorld(Vector3 worldPosition, ServerCommunication.AnnotationData annotation)
    {
        if (annotationPrefab == null)
        {
            Debug.LogError("❌ No annotation prefab assigned! Cannot place annotation.");
            return;
        }

        // Instantiate the annotation prefab
        GameObject annotationObject = Instantiate(annotationPrefab, worldPosition, Quaternion.identity);
        placedAnnotations.Add(annotationObject);
        annotationObject.name = $"Annotation_{annotation.id}";

        // Add annotation display component
        AnnotationDisplay display = annotationObject.GetComponent<AnnotationDisplay>();
        if (display == null)
        {
            display = annotationObject.AddComponent<AnnotationDisplay>();
        }
        display.SetAnnotationData(annotation);

        // Optional: Make annotations stand out visually
        Renderer renderer = annotationObject.GetComponent<Renderer>();
        if (renderer != null && renderer.material != null)
        {
            renderer.material.color = Color.red; // Make annotations red
        }

        if (debugAnnotations)
        {
            Debug.Log($"📍 Annotation {annotation.id} placed successfully");
            Debug.Log($"🎯 Represents user annotation from original camera view");
        }
    }

    // Public methods for external control
    public void ClearAllAnnotations()
    {
        foreach (GameObject annotation in placedAnnotations)
        {
            if (annotation != null) Destroy(annotation);
        }
        placedAnnotations.Clear();
        processedAnnotationIds.Clear();
        Debug.Log("🧹 Cleared all annotations");
    }

    public void LogAnnotationInfo()
    {
        Debug.Log($"📊 Annotation System Status:");
        Debug.Log($"   Placed annotations: {placedAnnotations.Count}");
        Debug.Log($"   Processed IDs: {string.Join(", ", processedAnnotationIds)}");
        Debug.Log($"   Polling enabled: {enableAnnotationSystem}");
    }
}

// Component to display annotation information
public class AnnotationDisplay : MonoBehaviour
{
    private ServerCommunication.AnnotationData annotationData;

    public void SetAnnotationData(ServerCommunication.AnnotationData data)
    {
        annotationData = data;
        CreateFloatingText();
    }

    private void CreateFloatingText()
    {
        // Create floating text above the annotation
        GameObject textObject = new GameObject("AnnotationText");
        textObject.transform.SetParent(this.transform);
        textObject.transform.localPosition = Vector3.up * 0.3f;

        TextMesh textMesh = textObject.AddComponent<TextMesh>();
        textMesh.text = $"Annotation {annotationData.id}\n({annotationData.center[0]}, {annotationData.center[1]})";
        textMesh.fontSize = 10;
        textMesh.color = Color.yellow;
        textMesh.anchor = TextAnchor.MiddleCenter;

        // Make text face the camera
        if (Camera.main != null)
        {
            textObject.transform.LookAt(Camera.main.transform);
            textObject.transform.Rotate(0, 180, 0);
        }
    }

    public ServerCommunication.AnnotationData GetAnnotationData() => annotationData;
}
