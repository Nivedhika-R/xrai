using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR.ARFoundation;

public class ARPlaceCube : MonoBehaviour
{
    [Header("AR Components")]
    [SerializeField] private ARRaycastManager raycastManager;
    [SerializeField] private ARPlaneManager planeManager;
    [SerializeField] private ARPointCloudManager pointCloudManager;

    [Header("Server Communication")]
    [SerializeField] private ServerCommunication serverCommunication;

    [Header("Annotation Settings")]
    [SerializeField] private GameObject annotationPrefab;
    [SerializeField] private bool enableAnnotationSystem = true;
    [Header("Annotation Visibility")]
    [SerializeField] private float annotationSize = 0.3f;       // BIGGER size for AR visibility
    [SerializeField] private bool alwaysFaceCamera = true;      // Always face camera
    [SerializeField] private float minDistanceFromCamera = 0.5f; // Minimum distance
    [SerializeField] private float maxDistanceFromCamera = 50f;  // Maximum distance

    [Header("Auto-Clear Settings")]
    [SerializeField] private bool clearPreviousAnnotations = true;
    [SerializeField] private bool clearOnNewAnnotation = true;
    [SerializeField] private bool clearOnTestAnnotation = true;

    [Header("Coordinate Calibration")]
    [SerializeField] private float xOffset = 0.1f;
    [SerializeField] private float yMultiplier = 1.2f;
    [SerializeField] private bool useCustomTransform = true;
    
    [Header("Sub-Pixel Fine-Tuning")]
    [SerializeField] private float microXOffset = 0.2f;
    [SerializeField] private float microYOffset = 0.2f;
    [SerializeField] private bool enableSubPixelTuning = true;

    [Header("Depth Calculation Settings")]
    [SerializeField] private bool usePoseMatrixDepth = true;     
    [SerializeField] private float maxRaycastDistance = 65f;    
    [SerializeField] private float defaultDepth = 1.5f;        // CLOSER default depth
    [SerializeField] private float minDepth = 0.3f;            // Minimum depth for close objects
    [SerializeField] private bool prioritizeCloseObjects = true; // Prefer close positioning
    [SerializeField] private bool debugDepthCalculation = true;

    [Header("Guaranteed Placement")]
    [SerializeField] private bool alwaysCreateAnnotation = true; 
    [SerializeField] private float fallbackDepth = 2.0f;

    [Header("Debug")]
    [SerializeField] private bool debugAnnotations = true;
    [SerializeField] private bool showDetailedLogs = true;
    [SerializeField] private bool createTestAnnotationOnStart = false;

    private List<GameObject> placedAnnotations = new List<GameObject>();
    private List<int> processedAnnotationIds = new List<int>();
    private Camera arCamera;

    void Start()
    {
        if (enableAnnotationSystem && serverCommunication != null)
        {
            InitializeAnnotationSystem();
        }

        InitializeARComponents();
        arCamera = Camera.main ?? FindObjectOfType<Camera>();

        Debug.Log("🎯 SIMPLE POSE MATRIX DEPTH CALCULATION");
        LogSystemStatus();

        if (createTestAnnotationOnStart)
        {
            Invoke(nameof(CreateTestAnnotation), 2f);
        }
    }

    private object CreateTestAnnotation()
    {
        throw new NotImplementedException();
    }

    void OnDestroy()
    {
        if (enableAnnotationSystem && serverCommunication != null)
        {
            serverCommunication.OnNewAnnotationReceived -= HandleNewAnnotation;
            serverCommunication.OnAnnotationError -= HandleAnnotationError;
        }
    }

    private void InitializeARComponents()
    {
        if (raycastManager == null)
            raycastManager = FindObjectOfType<ARRaycastManager>();
        if (planeManager == null)
            planeManager = FindObjectOfType<ARPlaneManager>();
        if (pointCloudManager == null)
            pointCloudManager = FindObjectOfType<ARPointCloudManager>();

        Debug.Log($"🔧 AR Components: Raycast={raycastManager != null}, Planes={planeManager != null}");
    }

    private void LogSystemStatus()
    {
        Debug.Log($"   📏 Size: {annotationSize}, Max raycast: {maxRaycastDistance}m");
        Debug.Log($"   🎯 Pose matrix depth: {usePoseMatrixDepth}");
        Debug.Log($"   📐 Default/Fallback depth: {defaultDepth}m / {fallbackDepth}m");
    }

    private void InitializeAnnotationSystem()
    {
        serverCommunication.OnNewAnnotationReceived += HandleNewAnnotation;
        serverCommunication.OnAnnotationError += HandleAnnotationError;
        serverCommunication.StartAnnotationPolling(2.0f);
        Debug.Log("🎯 Annotation system initialized");
    }

    private void ClearPreviousAnnotationsIfEnabled(string context = "")
    {
        if (!clearPreviousAnnotations) return;

        if (placedAnnotations.Count > 0)
        {
            Debug.Log($"🧹 Clearing {placedAnnotations.Count} annotations ({context})");
            ClearAllAnnotations();
        }
    }

    private void HandleNewAnnotation(ServerCommunication.AnnotationData annotation)
    {
        Debug.Log($"🎯 ANNOTATION ID {annotation.id}: ({annotation.center[0]}, {annotation.center[1]})");

        if (clearOnNewAnnotation)
            ClearPreviousAnnotationsIfEnabled("new annotation");

        Vector2 screenPoint = TransformToScreen(annotation.center[0], annotation.center[1]);
        Vector3 position = GetPositionWithDepthCalculation(screenPoint);

        CreateAnnotation(position, annotation);
        processedAnnotationIds.Add(annotation.id);
        serverCommunication.MarkAnnotationsAsProcessed(new int[] { annotation.id });

        Debug.Log($"✅ PLACED at {position} (depth: {Vector3.Distance(arCamera.transform.position, position):F2}m)");
    }

    /// <summary>
    /// MAIN METHOD: Get position with proper depth calculation
    /// Priority: 1) Pose matrix raycast, 2) AR raycast, 3) Guaranteed
    /// </summary>
    private Vector3 GetPositionWithDepthCalculation(Vector2 screenPoint)
    {
        // METHOD 1: Use pose matrix for depth calculation
        if (usePoseMatrixDepth)
        {
            Vector3 posePosition = CalculatePositionWithPoseMatrix(screenPoint);
            if (posePosition != Vector3.zero)
            {
                Debug.Log($"✅ POSE MATRIX: {posePosition}");
                return posePosition;
            }
        }

        // METHOD 2: AR raycast fallback
        Vector3 arPosition = TryARRaycast(screenPoint);
        if (arPosition != Vector3.zero)
        {
            Debug.Log($"✅ AR RAYCAST: {arPosition}");
            return arPosition;
        }

        // METHOD 3: Guaranteed positioning
        Vector3 guaranteedPos = GetGuaranteedPosition(screenPoint);
        Debug.Log($"✅ GUARANTEED: {guaranteedPos}");
        return guaranteedPos;
    }

    /// <summary>
    /// POSE MATRIX METHOD: Calculate position using current camera pose as server pose
    /// This demonstrates how it will work when server sends actual pose matrix
    /// </summary>
    private Vector3 CalculatePositionWithPoseMatrix(Vector2 screenPoint)
    {
        if (arCamera == null) return Vector3.zero;

        // Create ray from screen point (this simulates using server pose matrix)
        Ray ray = arCamera.ScreenPointToRay(new Vector3(screenPoint.x, screenPoint.y, 0));
        
        // Calculate depth using extended raycast with 50-70m range
        float calculatedDepth = CalculateDepthAlongRay(ray);
        
        // Create final position using calculated depth as Z coordinate
        Vector3 finalPosition = ray.origin + ray.direction.normalized * calculatedDepth;
        
        if (debugDepthCalculation)
        {
            Debug.Log($"   📏 DEPTH CALC: ray origin={ray.origin}, direction={ray.direction.normalized}");
            Debug.Log($"   📏 Calculated depth: {calculatedDepth:F3}m, Final pos: {finalPosition}");
        }
        
        return finalPosition;
    }

    /// <summary>
    /// Calculate depth along ray - PRIORITIZE CLOSE OBJECTS
    /// If you annotate something close, it should be placed close!
    /// </summary>
    private float CalculateDepthAlongRay(Ray ray)
    {
        float calculatedDepth = defaultDepth;
        bool hitFound = false;

        if (debugDepthCalculation)
        {
            Debug.Log($"   🔍 Calculating depth along ray (prioritizing close objects)");
        }

        // TRY 1: AR planes raycast - START WITH CLOSE RANGE FIRST
        if (raycastManager != null)
        {
            var hits = new List<ARRaycastHit>();
            Vector2 screenPos = arCamera.WorldToScreenPoint(ray.origin + ray.direction * 1f);
            
            if (raycastManager.Raycast(screenPos, hits, UnityEngine.XR.ARSubsystems.TrackableType.AllTypes))
            {
                // Sort hits by distance - closest first
                hits.Sort((a, b) => Vector3.Distance(ray.origin, a.pose.position).CompareTo(Vector3.Distance(ray.origin, b.pose.position)));
                
                calculatedDepth = Vector3.Distance(ray.origin, hits[0].pose.position);
                hitFound = true;
                
                if (debugDepthCalculation)
                {
                    Debug.Log($"      ✅ AR PLANE HIT at {calculatedDepth:F3}m (CLOSEST)");
                }
            }
        }

        // TRY 2: Physics raycast - PRIORITIZE CLOSE HITS
        if (!hitFound)
        {
            RaycastHit[] allHits = Physics.RaycastAll(ray, maxRaycastDistance);
            
            if (allHits.Length > 0)
            {
                // Sort by distance - closest first
                System.Array.Sort(allHits, (a, b) => a.distance.CompareTo(b.distance));
                
                calculatedDepth = allHits[0].distance;
                hitFound = true;
                
                if (debugDepthCalculation)
                {
                    Debug.Log($"      ✅ PHYSICS HIT at {calculatedDepth:F3}m on {allHits[0].collider.name} (CLOSEST of {allHits.Length} hits)");
                }
            }
        }

        // TRY 3: Point cloud sampling - LOOK FOR CLOSE POINTS FIRST
        if (!hitFound)
        {
            float pointDepth = SamplePointCloudDepthCloseFirst(ray);
            if (pointDepth > 0)
            {
                calculatedDepth = pointDepth;
                hitFound = true;
                
                if (debugDepthCalculation)
                {
                    Debug.Log($"      ✅ POINT CLOUD at {pointDepth:F3}m (CLOSE POINT)");
                }
            }
        }

        // FALLBACK: Use smart close depth for near annotations
        if (!hitFound)
        {
            if (prioritizeCloseObjects)
            {
                // For close annotations, use much closer depth
                calculatedDepth = CalculateCloseObjectDepth(ray);
                
                if (debugDepthCalculation)
                {
                    Debug.Log($"      🎯 CLOSE OBJECT DEPTH: {calculatedDepth:F3}m (prioritizing close placement)");
                }
            }
            else
            {
                calculatedDepth = CalculateIntelligentDepth(ray);
                
                if (debugDepthCalculation)
                {
                    Debug.Log($"      🎯 INTELLIGENT DEPTH: {calculatedDepth:F3}m");
                }
            }
        }

        // Clamp to valid range, but keep close objects close
        calculatedDepth = Mathf.Clamp(calculatedDepth, minDepth, maxRaycastDistance);
        
        if (debugDepthCalculation)
        {
            Debug.Log($"      📏 FINAL DEPTH: {calculatedDepth:F3}m (hit found: {hitFound})");
        }
        
        return calculatedDepth;
    }

    /// <summary>
    /// Sample point cloud prioritizing close points first
    /// </summary>
    private float SamplePointCloudDepthCloseFirst(Ray ray)
    {
        if (pointCloudManager == null) return 0f;

        List<float> nearbyDepths = new List<float>();

        foreach (var pointCloud in pointCloudManager.trackables)
        {
            if (pointCloud.positions.HasValue)
            {
                var positions = pointCloud.positions.Value;
                
                for (int i = 0; i < positions.Length; i += 3) // Sample more points for accuracy
                {
                    Vector3 worldPoint = pointCloud.transform.TransformPoint(positions[i]);
                    float distanceToRay = Vector3.Cross(ray.direction, worldPoint - ray.origin).magnitude;
                    
                    if (distanceToRay < 0.5f) // Point is close to ray
                    {
                        float depth = Vector3.Distance(ray.origin, worldPoint);
                        if (depth > minDepth && depth <= maxRaycastDistance)
                        {
                            nearbyDepths.Add(depth);
                        }
                    }
                }
            }
        }

        if (nearbyDepths.Count > 0)
        {
            // Sort and return the closest depth
            nearbyDepths.Sort();
            return nearbyDepths[0]; // Return closest point
        }

        return 0f;
    }

    /// <summary>
    /// Calculate depth specifically for close objects that were annotated
    /// If someone annotates something close, place it close!
    /// </summary>
    private float CalculateCloseObjectDepth(Ray ray)
    {
        // For close annotations, use much shorter depths
        float baseCloseDepth = defaultDepth * 0.7f; // 30% closer than default
        
        // Factor based on ray direction - straight ahead = closer, angled = slightly further
        float straightAheadFactor = Vector3.Dot(ray.direction, arCamera.transform.forward);
        float depthAdjustment = 1.0f - (straightAheadFactor * 0.3f); // More straight = closer
        
        float closeDepth = baseCloseDepth * depthAdjustment;
        
        // Ensure it's in close range
        closeDepth = Mathf.Clamp(closeDepth, minDepth, defaultDepth * 1.5f);
        
        if (debugDepthCalculation)
        {
            Debug.Log($"      📏 CLOSE OBJECT CALCULATION:");
            Debug.Log($"         Base close depth: {baseCloseDepth:F3}m");
            Debug.Log($"         Straight ahead factor: {straightAheadFactor:F3}");
            Debug.Log($"         Final close depth: {closeDepth:F3}m");
        }
        
        return closeDepth;
    }

    /// <summary>
    /// Calculate intelligent depth when no surfaces detected - PREFER CLOSE DEPTHS
    /// </summary>
    private float CalculateIntelligentDepth(Ray ray)
    {
        Vector3 cameraPos = ray.origin;
        Vector3 rayDir = ray.direction;
        
        // Start with closer base depth
        float baseDepth = defaultDepth * 0.8f; // 20% closer than default
        
        // Factor 1: Ray direction - straight ahead should be closer
        float forwardDot = Vector3.Dot(rayDir, arCamera.transform.forward);
        float directionFactor = 1.0f - (forwardDot * 0.4f); // More forward = closer
        
        // Factor 2: Screen center proximity - center annotations often closer
        // This is estimated since we don't have exact screen coordinates here
        float centerFactor = 0.9f; // Assume annotations are somewhat central
        
        // Factor 3: Camera height - lower camera = closer objects typical
        float heightFactor = 1.0f - (Mathf.Abs(cameraPos.y) * 0.1f);
        heightFactor = Mathf.Clamp(heightFactor, 0.6f, 1.0f);
        
        float estimatedDepth = baseDepth * directionFactor * centerFactor * heightFactor;
        
        // Clamp to close-object range
        estimatedDepth = Mathf.Clamp(estimatedDepth, minDepth, defaultDepth * 2f);
        
        if (debugDepthCalculation)
        {
            Debug.Log($"      📊 CLOSE DEPTH FACTORS:");
            Debug.Log($"         Direction factor: {directionFactor:F3} (forward dot: {forwardDot:F3})");
            Debug.Log($"         Center factor: {centerFactor:F3}");
            Debug.Log($"         Height factor: {heightFactor:F3}");
            Debug.Log($"         Final estimated: {estimatedDepth:F3}m");
        }
        
        return estimatedDepth;
    }

    private Vector2 TransformToScreen(float annotationX, float annotationY)
    {
        if (arCamera == null)
            return new Vector2(annotationX, annotationY);

        float normalizedX = annotationX / 640f;
        float normalizedY = annotationY / 480f;

        if (useCustomTransform)
        {
            normalizedX = normalizedX + xOffset;
            normalizedY = yMultiplier - normalizedY;
            
            if (enableSubPixelTuning)
            {
                normalizedX += microXOffset;
                normalizedY += microYOffset;
            }
        }
        else
        {
            normalizedY = 1.0f - normalizedY;
        }

        float screenX = normalizedX * arCamera.pixelWidth;
        float screenY = normalizedY * arCamera.pixelHeight;

        screenX = Mathf.Clamp(screenX, 0f, arCamera.pixelWidth);
        screenY = Mathf.Clamp(screenY, 0f, arCamera.pixelHeight);

        return new Vector2(screenX, screenY);
    }

    private Vector3 TryARRaycast(Vector2 screenPoint)
    {
        if (raycastManager == null) return Vector3.zero;

        var hits = new List<ARRaycastHit>();
        
        if (raycastManager.Raycast(screenPoint, hits, UnityEngine.XR.ARSubsystems.TrackableType.AllTypes))
        {
            return hits[0].pose.position;
        }

        return Vector3.zero;
    }

    private Vector3 GetGuaranteedPosition(Vector2 screenPoint)
    {
        if (arCamera == null)
            return Vector3.forward * fallbackDepth;

        Ray ray = arCamera.ScreenPointToRay(new Vector3(screenPoint.x, screenPoint.y, 0));
        return ray.origin + ray.direction.normalized * fallbackDepth;
    }

    private void CreateAnnotation(Vector3 position, ServerCommunication.AnnotationData annotation)
    {
        // Ensure annotation is at reasonable distance from camera
        float distanceFromCamera = Vector3.Distance(arCamera.transform.position, position);
        
        if (distanceFromCamera < minDistanceFromCamera)
        {
            // Move annotation to minimum distance
            Vector3 direction = (position - arCamera.transform.position).normalized;
            position = arCamera.transform.position + direction * minDistanceFromCamera;
            Debug.Log($"📏 Moved annotation to minimum distance: {minDistanceFromCamera}m");
        }
        else if (distanceFromCamera > maxDistanceFromCamera)
        {
            // Move annotation to maximum distance
            Vector3 direction = (position - arCamera.transform.position).normalized;
            position = arCamera.transform.position + direction * maxDistanceFromCamera;
            Debug.Log($"📏 Moved annotation to maximum distance: {maxDistanceFromCamera}m");
        }

        GameObject annotationObject = CreateVisibleARAnnotation(position);
        annotationObject.transform.localScale = Vector3.one * annotationSize;
        placedAnnotations.Add(annotationObject);
        annotationObject.name = $"Annotation_{annotation.id}";

        Debug.Log($"📍 CREATED: {annotationObject.name} at {position} (distance: {Vector3.Distance(arCamera.transform.position, position):F2}m)");
        
        // Immediately verify it's visible
        VerifyAnnotationInCameraView(annotationObject);
    }

    /// <summary>
    /// Create annotation optimized for AR visibility - MUCH BIGGER AND BRIGHTER
    /// </summary>
    private GameObject CreateVisibleARAnnotation(Vector3 position)
    {
        // Use Sphere for better 3D visibility in AR
        GameObject annotationObject = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        annotationObject.transform.position = position;
        annotationObject.name = "ARAnnotation";

        // Remove collider
        Collider sphereCollider = annotationObject.GetComponent<Collider>();
        if (sphereCollider != null)
            Destroy(sphereCollider);

        // Create VERY BRIGHT, VISIBLE material
        Renderer renderer = annotationObject.GetComponent<Renderer>();
        if (renderer != null)
        {
            Material annotationMaterial = new Material(Shader.Find("Standard"));
            
            // BRIGHT RED with high emission
            annotationMaterial.color = Color.red;
            annotationMaterial.SetFloat("_Metallic", 0f);
            annotationMaterial.SetFloat("_Glossiness", 0.8f);
            
            // STRONG EMISSION - makes it glow and always visible
            annotationMaterial.EnableKeyword("_EMISSION");
            annotationMaterial.SetColor("_EmissionColor", Color.red * 2.0f); // VERY BRIGHT
            
            // Ensure it renders on top
            annotationMaterial.renderQueue = 3000;
            
            renderer.material = annotationMaterial;
            
            Debug.Log($"   🔴 Created BRIGHT GLOWING sphere annotation");
        }

        // Always face camera
        if (alwaysFaceCamera)
        {
            annotationObject.AddComponent<SimpleBillboard>();
        }

        return annotationObject;
    }

    [ContextMenu("Create CLOSE Test Annotation")]
    public void CreateCloseTestAnnotation()
    {
        Debug.Log("🧪 CREATING CLOSE TEST ANNOTATION (should be 1-2m away)");

        if (arCamera == null)
        {
            Debug.LogError("❌ No camera");
            return;
        }

        if (clearOnTestAnnotation)
            ClearPreviousAnnotationsIfEnabled("close test");

        // Force a close position - 1.5m directly in front of camera
        Vector3 closePosition = arCamera.transform.position + arCamera.transform.forward * 1.5f;
        
        var testAnnotation = new ServerCommunication.AnnotationData
        {
            id = 999,
            center = new int[] { 320, 240 } // Center
        };
        
        // Create annotation at close position
        GameObject closeAnnotation = CreateVisibleARAnnotation(closePosition);
        closeAnnotation.transform.localScale = Vector3.one * (annotationSize * 2f); // EXTRA LARGE
        closeAnnotation.name = "CLOSE_TEST_ANNOTATION";
        placedAnnotations.Add(closeAnnotation);
        
        Debug.Log($"🧪 CLOSE test annotation created at {closePosition}");
        Debug.Log($"📏 Distance from camera: {Vector3.Distance(arCamera.transform.position, closePosition):F2}m");
        Debug.Log($"📏 Scale: {closeAnnotation.transform.localScale.x:F2}");
        Debug.Log($"👁️ This should be a LARGE RED GLOWING SPHERE 1.5m in front of you!");
        
        VerifyAnnotationInCameraView(closeAnnotation);
    }

    [ContextMenu("Fix Depth for Close Objects")]
    public void FixDepthForCloseObjects()
    {
        Debug.Log("🔧 FIXING DEPTH CALCULATION FOR CLOSE OBJECTS");
        
        // Update settings to prioritize close objects
        prioritizeCloseObjects = true;
        defaultDepth = 1.2f;  // Much closer default
        minDepth = 0.3f;
        
        Debug.Log($"✅ Updated settings:");
        Debug.Log($"   Default depth: {defaultDepth}m (was 2.0m)");
        Debug.Log($"   Min depth: {minDepth}m");
        Debug.Log($"   Prioritize close: {prioritizeCloseObjects}");
        
        // Clear existing annotations and create a close test
        ClearAllAnnotations();
        CreateCloseTestAnnotation();
    }

    /// <summary>
    /// Verify annotation is actually visible in camera view
    /// </summary>
    private void VerifyAnnotationInCameraView(GameObject annotation)
    {
        if (annotation == null || arCamera == null) return;

        Vector3 annotationPos = annotation.transform.position;
        Vector3 screenPos = arCamera.WorldToScreenPoint(annotationPos);
        
        bool inFrontOfCamera = screenPos.z > 0;
        bool inScreenBounds = screenPos.x >= 0 && screenPos.x <= arCamera.pixelWidth && 
                             screenPos.y >= 0 && screenPos.y <= arCamera.pixelHeight;
        float distance = Vector3.Distance(arCamera.transform.position, annotationPos);
        
        Debug.Log($"   👁️ VISIBILITY CHECK for {annotation.name}:");
        Debug.Log($"      World position: {annotationPos}");
        Debug.Log($"      Screen position: ({screenPos.x:F1}, {screenPos.y:F1}, depth: {screenPos.z:F2})");
        Debug.Log($"      In front of camera: {(inFrontOfCamera ? "✅ YES" : "❌ NO")}");
        Debug.Log($"      In screen bounds: {(inScreenBounds ? "✅ YES" : "❌ NO")}");
        Debug.Log($"      Distance: {distance:F2}m");
        Debug.Log($"      Scale: {annotation.transform.localScale.x:F3}");
        
        if (!inFrontOfCamera)
        {
            Debug.LogWarning($"⚠️ {annotation.name} is BEHIND the camera!");
        }
        
        if (!inScreenBounds)
        {
            Debug.LogWarning($"⚠️ {annotation.name} is OUTSIDE screen bounds!");
        }
        
        if (distance < 0.1f)
        {
            Debug.LogWarning($"⚠️ {annotation.name} is TOO CLOSE to camera!");
        }
        
        if (inFrontOfCamera && inScreenBounds && distance > 0.1f)
        {
            Debug.Log($"   ✅ {annotation.name} should be VISIBLE in AR view!");
        }
    }

    private GameObject CreateVisibleCircle(Vector3 position)
    {
        GameObject circleObject = GameObject.CreatePrimitive(PrimitiveType.Quad);
        circleObject.transform.position = position;
        circleObject.name = "CircleAnnotation";

        Collider quadCollider = circleObject.GetComponent<Collider>();
        if (quadCollider != null)
            Destroy(quadCollider);

        Renderer renderer = circleObject.GetComponent<Renderer>();
        if (renderer != null)
        {
            Texture2D circleTexture = CreateCircleTexture();
            Material circleMaterial = new Material(Shader.Find("Sprites/Default"));
            circleMaterial.mainTexture = circleTexture;
            circleMaterial.color = new Color(1f, 0f, 0f, 0.5f);
            
            circleMaterial.SetFloat("_Mode", 3);
            circleMaterial.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
            circleMaterial.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
            circleMaterial.SetInt("_ZWrite", 0);
            circleMaterial.EnableKeyword("_ALPHABLEND_ON");
            circleMaterial.renderQueue = 3000;
            
            renderer.material = circleMaterial;
        }

        circleObject.AddComponent<SimpleBillboard>();
        return circleObject;
    }

    private Texture2D CreateCircleTexture()
    {
        int size = 128;
        Texture2D texture = new Texture2D(size, size, TextureFormat.RGBA32, false);
        Color[] pixels = new Color[size * size];

        Vector2 center = new Vector2(size / 2f, size / 2f);
        float outerRadius = size / 2f - 2f;
        float innerRadius = outerRadius - 4f;

        for (int y = 0; y < size; y++)
        {
            for (int x = 0; x < size; x++)
            {
                Vector2 pos = new Vector2(x, y);
                float distance = Vector2.Distance(pos, center);
                
                if (distance <= innerRadius)
                    pixels[y * size + x] = new Color(1f, 0f, 0f, 0.1f);
                else if (distance <= outerRadius)
                    pixels[y * size + x] = new Color(1f, 0f, 0f, 1f);
                else
                    pixels[y * size + x] = Color.clear;
            }
        }

        texture.SetPixels(pixels);
        texture.Apply();
        texture.filterMode = FilterMode.Bilinear;
        return texture;
    }

    public class SimpleBillboard : MonoBehaviour
    {
        private Camera targetCamera;

        void Start()
        {
            targetCamera = Camera.main ?? FindObjectOfType<Camera>();
        }

        void Update()
        {
            if (targetCamera != null)
            {
                Vector3 lookDirection = targetCamera.transform.position - transform.position;
                transform.rotation = Quaternion.LookRotation(-lookDirection);
            }
        }
    }

    private void HandleAnnotationError(string error)
    {
        Debug.LogWarning($"⚠️ Annotation Error: {error}");
    }

    public void ClearAllAnnotations()
    {
        foreach (var annotation in placedAnnotations)
        {
            if (annotation != null)
                Destroy(annotation);
        }
        placedAnnotations.Clear();
        processedAnnotationIds.Clear();
    }

    // ==========================================
    // SERVER POSE MATRIX INTEGRATION METHODS
    // ==========================================

    /// <summary>
    /// When server sends pose matrix data, use this method for EXACT positioning
    /// Call this when ServerCommunication.AnnotationData includes cameraToWorldMatrix
    /// </summary>
    public void CreateAnnotationWithServerPose(ServerCommunication.AnnotationData annotation, 
                                              float[] cameraToWorldMatrix, 
                                              float[] intrinsics = null)
    {
        if (cameraToWorldMatrix == null || cameraToWorldMatrix.Length != 16)
        {
            Debug.LogWarning("⚠️ Invalid server pose matrix - using fallback");
            HandleNewAnnotation(annotation);
            return;
        }

        Debug.Log($"🎯 ANNOTATION WITH SERVER POSE: ID {annotation.id}");

        if (clearOnNewAnnotation)
            ClearPreviousAnnotationsIfEnabled("server pose annotation");

        // Convert server matrix to Unity format
        Matrix4x4 serverPose = ArrayToMatrix4x4(cameraToWorldMatrix);
        
        // Extract server camera position and rotation
        Vector3 serverCameraPos = new Vector3(serverPose.m03, serverPose.m13, serverPose.m23);
        Quaternion serverCameraRot = serverPose.rotation;

        // Transform coordinates and create ray
        Vector2 screenPoint = TransformToScreen(annotation.center[0], annotation.center[1]);
        Ray serverRay = CreateRayFromServerPose(screenPoint, serverCameraPos, serverCameraRot, intrinsics);
        
        // Calculate depth with extended range
        float depth = CalculateDepthAlongRay(serverRay);
        
        // Final position using calculated depth as Z coordinate
        Vector3 finalPosition = serverRay.origin + serverRay.direction.normalized * depth;

        CreateAnnotation(finalPosition, annotation);
        
        Debug.Log($"✅ SERVER POSE ANNOTATION: depth={depth:F2}m, position={finalPosition}");
    }

    /// <summary>
    /// Convert float array to Matrix4x4
    /// </summary>
    private Matrix4x4 ArrayToMatrix4x4(float[] array)
    {
        Matrix4x4 matrix = Matrix4x4.identity;
        if (array.Length == 16)
        {
            matrix.m00 = array[0];  matrix.m01 = array[1];  matrix.m02 = array[2];  matrix.m03 = array[3];
            matrix.m10 = array[4];  matrix.m11 = array[5];  matrix.m12 = array[6];  matrix.m13 = array[7];
            matrix.m20 = array[8];  matrix.m21 = array[9];  matrix.m22 = array[10]; matrix.m23 = array[11];
            matrix.m30 = array[12]; matrix.m31 = array[13]; matrix.m32 = array[14]; matrix.m33 = array[15];
        }
        return matrix;
    }

    /// <summary>
    /// Create ray from server pose data
    /// </summary>
    private Ray CreateRayFromServerPose(Vector2 screenPoint, Vector3 cameraPos, Quaternion cameraRot, float[] intrinsics)
    {
        Vector3 rayDirection;
        
        if (intrinsics != null && intrinsics.Length >= 4)
        {
            // Use camera intrinsics for precise ray
            float fx = intrinsics[0];
            float fy = intrinsics[5]; 
            float cx = intrinsics[2];
            float cy = intrinsics[6];
            
            float x = (screenPoint.x - cx) / fx;
            float y = (screenPoint.y - cy) / fy;
            rayDirection = new Vector3(x, y, 1f).normalized;
        }
        else
        {
            // Simple screen-to-ray conversion
            float normalizedX = (screenPoint.x / arCamera.pixelWidth) * 2.0f - 1.0f;
            float normalizedY = (screenPoint.y / arCamera.pixelHeight) * 2.0f - 1.0f;
            rayDirection = new Vector3(normalizedX, normalizedY, 1f).normalized;
        }
        
        // Transform ray direction by camera rotation
        rayDirection = cameraRot * rayDirection;
        
        return new Ray(cameraPos, rayDirection);
    }

    // ==========================================
    // TEST METHODS
    // ==========================================

    [ContextMenu("Create Visible Test Annotation")]
    public void CreateVisibleTestAnnotation()
    {
        Debug.Log("🧪 CREATING HIGHLY VISIBLE TEST ANNOTATION");

        if (arCamera == null)
        {
            Debug.LogError("❌ No camera");
            return;
        }

        if (clearOnTestAnnotation)
            ClearPreviousAnnotationsIfEnabled("visible test");

        // Place annotation directly in front of camera at reasonable distance
        Vector3 cameraForward = arCamera.transform.forward;
        Vector3 testPosition = arCamera.transform.position + cameraForward * 3.0f; // 3m in front
        
        var testAnnotation = new ServerCommunication.AnnotationData
        {
            id = 999,
            center = new int[] { 320, 240 } // Center
        };
        
        CreateAnnotation(testPosition, testAnnotation);
        
        Debug.Log($"🧪 VISIBLE test annotation created 3m in front of camera");
        Debug.Log($"👁️ This should be CLEARLY VISIBLE in your AR view!");
        Debug.Log($"📍 Position: {testPosition}");
    }

    [ContextMenu("Create Multiple Visible Annotations")]
    public void CreateMultipleVisibleAnnotations()
    {
        Debug.Log("🧪 CREATING MULTIPLE VISIBLE ANNOTATIONS");

        if (arCamera == null)
        {
            Debug.LogError("❌ No camera");
            return;
        }

        if (clearOnTestAnnotation)
            ClearPreviousAnnotationsIfEnabled("multiple visible test");

        Vector3 cameraPos = arCamera.transform.position;
        Vector3 cameraForward = arCamera.transform.forward;
        Vector3 cameraRight = arCamera.transform.right;
        Vector3 cameraUp = arCamera.transform.up;

        // Create annotations at different positions around the camera
        Vector3[] testPositions = new Vector3[]
        {
            cameraPos + cameraForward * 2f,                           // Center front
            cameraPos + cameraForward * 2f + cameraRight * 1f,        // Right
            cameraPos + cameraForward * 2f - cameraRight * 1f,        // Left  
            cameraPos + cameraForward * 2f + cameraUp * 0.5f,         // Up
            cameraPos + cameraForward * 2f - cameraUp * 0.5f,         // Down
            cameraPos + cameraForward * 4f,                           // Further back
        };

        string[] names = { "Center", "Right", "Left", "Up", "Down", "Far" };

        for (int i = 0; i < testPositions.Length; i++)
        {
            var testAnnotation = new ServerCommunication.AnnotationData
            {
                id = 700 + i,
                center = new int[] { 320 + (i * 20), 240 } // Slightly offset centers
            };
            
            CreateAnnotation(testPositions[i], testAnnotation);
            Debug.Log($"📍 {names[i]} annotation at {testPositions[i]}");
        }
        
        Debug.Log($"🧪 Created {testPositions.Length} visible test annotations");
        Debug.Log($"👁️ You should see {testPositions.Length} RED SPHERES in your AR view!");
    }

    [ContextMenu("Test Maximum Range")]
    public void TestMaximumRange()
    {
        Debug.Log($"🚀 TESTING MAXIMUM RANGE ({maxRaycastDistance}m)");

        if (clearOnTestAnnotation)
            ClearPreviousAnnotationsIfEnabled("max range test");

        // Create annotation at different angles to test range
        Vector2[] testPoints = {
            new Vector2(320f, 120f),  // Top center - upward angle
            new Vector2(320f, 240f),  // Center - straight ahead  
            new Vector2(320f, 360f),  // Bottom center - downward angle
        };

        for (int i = 0; i < testPoints.Length; i++)
        {
            Vector2 screenPoint = TransformToScreen(testPoints[i].x, testPoints[i].y);
            Ray ray = arCamera.ScreenPointToRay(new Vector3(screenPoint.x, screenPoint.y, 0));
            
            float depth = CalculateDepthAlongRay(ray);
            Vector3 position = ray.origin + ray.direction.normalized * depth;
            
            var annotation = new ServerCommunication.AnnotationData
            {
                id = 800 + i,
                center = new int[] { (int)testPoints[i].x, (int)testPoints[i].y }
            };
            
            CreateAnnotation(position, annotation);
            Debug.Log($"🚀 Test {i+1}: Depth={depth:F2}m, Position={position}");
        }
    }

    [ContextMenu("Debug System")]
    public void DebugSystem()
    {
        Debug.Log("🔍 SYSTEM DEBUG:");
        Debug.Log($"   Pose matrix depth: {usePoseMatrixDepth}");
        Debug.Log($"   Max raycast distance: {maxRaycastDistance}m");
        Debug.Log($"   Current annotations: {placedAnnotations.Count}");
        Debug.Log($"   AR Components: Raycast={raycastManager != null}, Planes={planeManager != null}");
        
        if (arCamera != null)
        {
            Debug.Log($"   Camera: {arCamera.pixelWidth}x{arCamera.pixelHeight}, FOV={arCamera.fieldOfView:F1}°");
        }
    }
}

// // using System.Collections;
// // using System.Collections.Generic;
// // using UnityEngine;
// // using UnityEngine.XR.ARFoundation;

// // public class ARPlaceCube : MonoBehaviour
// // {
// //     [Header("AR Components")]
// //     [SerializeField] private ARRaycastManager raycastManager;
// //     [SerializeField] private ARPlaneManager planeManager;
// //     [SerializeField] private ARPointCloudManager pointCloudManager;

// //     [Header("Server Communication")]
// //     [SerializeField] private ServerCommunication serverCommunication;

// //     [Header("Annotation Settings")]
// //     [SerializeField] private GameObject annotationPrefab;
// //     [SerializeField] private bool enableAnnotationSystem = true;
// //     [SerializeField] private float annotationSize = 0.05f; // UNIFIED SIZE FOR ALL ANNOTATIONS

// //     [Header("Guaranteed Placement - ALWAYS CREATE ANNOTATIONS")]
// //     [SerializeField] private bool alwaysCreateAnnotation = true; 
// //     [SerializeField] private float fallbackDepth = 2.0f;
// //     [SerializeField] private bool forceGuaranteedPlacement = true;
// //     [SerializeField] private bool preferHitTestWhenAvailable = true;

// //     [Header("Debug")]
// //     [SerializeField] private bool debugAnnotations = true;
// //     [SerializeField] private bool showDetailedLogs = true;
// //     [SerializeField] private bool createTestAnnotationOnStart = false;

// //     private List<GameObject> placedAnnotations = new List<GameObject>();
// //     private List<int> processedAnnotationIds = new List<int>();
// //     private Camera arCamera;

// //     void Start()
// //     {
// //         if (enableAnnotationSystem && serverCommunication != null)
// //         {
// //             InitializeAnnotationSystem();
// //         }

// //         InitializeARComponents();
// //         arCamera = Camera.main ?? FindObjectOfType<Camera>();

// //         Debug.Log("🎯 SIMPLE ANNOTATION PLACEMENT SYSTEM - NO COMPLEX TRANSFORMS");
// //         LogSystemStatus();
// //         CheckPrefabAssignment();

// //         if (createTestAnnotationOnStart)
// //         {
// //             Debug.Log("🧪 Creating test annotation on start");
// //             Invoke(nameof(CreateTestAnnotation), 2f);
// //         }
// //     }

// //     void OnDestroy()
// //     {
// //         if (enableAnnotationSystem && serverCommunication != null)
// //         {
// //             serverCommunication.OnNewAnnotationReceived -= HandleNewAnnotation;
// //             serverCommunication.OnAnnotationError -= HandleAnnotationError;
// //         }
// //     }

// //     private void InitializeARComponents()
// //     {
// //         if (raycastManager == null)
// //             raycastManager = FindObjectOfType<ARRaycastManager>();

// //         if (planeManager == null)
// //             planeManager = FindObjectOfType<ARPlaneManager>();

// //         if (pointCloudManager == null)
// //             pointCloudManager = FindObjectOfType<ARPointCloudManager>();

// //         Debug.Log($"🔧 AR COMPONENTS:");
// //         Debug.Log($"   Raycast Manager: {(raycastManager != null ? "✅" : "❌")}");
// //         Debug.Log($"   Plane Manager: {(planeManager != null ? "✅" : "❌")}");
// //         Debug.Log($"   Point Cloud Manager: {(pointCloudManager != null ? "✅" : "❌")}");
// //     }

// //     private void LogSystemStatus()
// //     {
// //         Debug.Log($"   📏 UNIFORM SIZE: {annotationSize}");
// //         Debug.Log($"   🎯 GUARANTEED PLACEMENT: ENABLED");
// //         Debug.Log($"   📐 Fallback depth: {fallbackDepth}m");
// //         Debug.Log($"   🖼️ SIMPLE FORMULA: (x/640, (1-y/480)) × screen");
// //         Debug.Log($"   📱 Screen: {(arCamera ? $"{arCamera.pixelWidth}x{arCamera.pixelHeight}" : "Unknown")}");

// //         if (arCamera != null)
// //         {
// //             float scaleX = arCamera.pixelWidth / 640f;
// //             float scaleY = arCamera.pixelHeight / 480f;
// //             Debug.Log($"   📐 Scale: X={scaleX:F3}, Y={scaleY:F3}");
// //         }
// //     }

// //     private void CheckPrefabAssignment()
// //     {
// //         if (annotationPrefab == null)
// //         {
// //             Debug.LogWarning("⚠️ No prefab - will create spheres");
// //         }
// //         else
// //         {
// //             Debug.Log($"✅ Prefab: {annotationPrefab.name}");
// //         }
// //     }

// //     private void InitializeAnnotationSystem()
// //     {
// //         serverCommunication.OnNewAnnotationReceived += HandleNewAnnotation;
// //         serverCommunication.OnAnnotationError += HandleAnnotationError;
// //         serverCommunication.StartAnnotationPolling(2.0f);
// //         Debug.Log("🎯 Annotation system initialized");
// //     }

// //     private void HandleNewAnnotation(ServerCommunication.AnnotationData annotation)
// //     {
// //         Debug.Log("═══════════════════════════════════════");
// //         Debug.Log($"🎯 ANNOTATION: ID {annotation.id}");

// //         float annotationX = annotation.center[0];
// //         float annotationY = annotation.center[1];

// //         Debug.Log($"📍 Input: ({annotationX}, {annotationY}) in 640x480");

// //         // SIMPLE COORDINATE TRANSFORMATION
// //         Vector2 screenPoint = TransformToScreen(annotationX, annotationY);
// //         Debug.Log($"📍 Screen: ({screenPoint.x:F1}, {screenPoint.y:F1})");

// //         // GUARANTEED POSITION
// //         Vector3 position = GetGuaranteedPosition(screenPoint);

// //         // CREATE ANNOTATION
// //         CreateAnnotation(position, annotation);

// //         processedAnnotationIds.Add(annotation.id);
// //         serverCommunication.MarkAnnotationsAsProcessed(new int[] { annotation.id });

// //         Debug.Log($"✅ ANNOTATION PLACED at {position}");
// //         Debug.Log($"📍 Total annotations: {placedAnnotations.Count}");
// //         Debug.Log("═══════════════════════════════════════");
// //     }

// //     /// <summary>
// //     /// SIMPLE: Just divide by 640x480 and multiply by screen resolution
// //     /// </summary>
// //     private Vector2 TransformToScreen(float annotationX, float annotationY)
// //     {
// //         if (arCamera == null)
// //         {
// //             return new Vector2(annotationX, annotationY);
// //         }

// //         // Step 1: Normalize to 0-1
// //         float normalizedX = annotationX / 640f;
// //         float normalizedY = annotationY / 480f;

// //         // Step 2: Flip Y (annotation Y=0 is top, Unity Y=0 is bottom)
// //         normalizedX = normalizedX + 0.10f;
// //         normalizedY = 1.1f - normalizedY;

// //         // Step 3: Scale to screen
// //         float screenX = normalizedX * arCamera.pixelWidth;
// //         float screenY = normalizedY * arCamera.pixelHeight;

// //         // Step 4: Clamp to bounds
// //         screenX = Mathf.Clamp(screenX, 0f, arCamera.pixelWidth - 1f);
// //         screenY = Mathf.Clamp(screenY, 0f, arCamera.pixelHeight - 1f);

// //         if (showDetailedLogs)
// //         {
// //             Debug.Log($"   Transform: ({annotationX}, {annotationY}) → norm({normalizedX:F3}, {normalizedY:F3}) → screen({screenX:F1}, {screenY:F1})");
// //         }

// //         return new Vector2(screenX, screenY);
// //     }

// //     /// <summary>
// //     /// GUARANTEED: Always returns a valid position
// //     /// </summary>
// //     private Vector3 GetGuaranteedPosition(Vector2 screenPoint)
// //     {
// //         Debug.Log($"🔍 Getting position for screen ({screenPoint.x:F1}, {screenPoint.y:F1})");

// //         // TRY 1: AR Raycast
// //         if (preferHitTestWhenAvailable && raycastManager != null)
// //         {
// //             Vector3 arHit = TryARRaycast(screenPoint);
// //             if (arHit != Vector3.zero)
// //             {
// //                 Debug.Log($"✅ AR Hit: {arHit}");
// //                 return arHit;
// //             }
// //         }

// //         // TRY 2: Physics Raycast
// //         if (preferHitTestWhenAvailable)
// //         {
// //             Vector3 physicsHit = TryPhysicsRaycast(screenPoint);
// //             if (physicsHit != Vector3.zero)
// //             {
// //                 Debug.Log($"✅ Physics Hit: {physicsHit}");
// //                 return physicsHit;
// //             }
// //         }

// //         // GUARANTEED: Camera ray at fixed depth
// //         Vector3 guaranteedPos = GetCameraRayPosition(screenPoint);
// //         Debug.Log($"✅ Guaranteed: {guaranteedPos}");
// //         return guaranteedPos;
// //     }

// //     /// <summary>
// //     /// Try AR raycast - return zero if no hit
// //     /// </summary>
// //     private Vector3 TryARRaycast(Vector2 screenPoint)
// //     {
// //         if (raycastManager == null) return Vector3.zero;

// //         var hits = new List<ARRaycastHit>();

// //         if (raycastManager.Raycast(screenPoint, hits, UnityEngine.XR.ARSubsystems.TrackableType.AllTypes))
// //         {
// //             float distance = Vector3.Distance(arCamera.transform.position, hits[0].pose.position);
// //             Debug.Log($"   AR raycast HIT at {distance:F2}m");
// //             return hits[0].pose.position;
// //         }

// //         Debug.Log($"   AR raycast MISS");
// //         return Vector3.zero;
// //     }

// //     /// <summary>
// //     /// Try physics raycast - return zero if no hit
// //     /// </summary>
// //     private Vector3 TryPhysicsRaycast(Vector2 screenPoint)
// //     {
// //         if (arCamera == null) return Vector3.zero;

// //         Ray ray = arCamera.ScreenPointToRay(new Vector3(screenPoint.x, screenPoint.y, 0));
// //         RaycastHit hit;

// //         if (Physics.Raycast(ray, out hit, 20f))
// //         {
// //             Debug.Log($"   Physics raycast HIT {hit.collider.name} at {hit.distance:F2}m");
// //             return hit.point;
// //         }

// //         Debug.Log($"   Physics raycast MISS");
// //         return Vector3.zero;
// //     }

// //     /// <summary>
// //     /// GUARANTEED: Use camera ray at fixed depth - ALWAYS works
// //     /// </summary>
// //     private Vector3 GetCameraRayPosition(Vector2 screenPoint)
// //     {
// //         if (arCamera == null)
// //         {
// //             // No camera - place in world forward
// //             return Vector3.forward * fallbackDepth;
// //         }

// //         // Use Unity's screen-to-ray for perfect coordinate mapping
// //         Ray ray = arCamera.ScreenPointToRay(new Vector3(screenPoint.x, screenPoint.y, 0));
// //         Vector3 position = ray.origin + ray.direction.normalized * fallbackDepth;

// //         if (showDetailedLogs)
// //         {
// //             Debug.Log($"   Camera ray: origin={ray.origin}, dir={ray.direction.normalized}, pos={position}");
// //         }

// //         return position;
// //     }

// //     /// <summary>
// //     /// Create annotation with uniform size
// //     /// </summary>
// //     private void CreateAnnotation(Vector3 position, ServerCommunication.AnnotationData annotation)
// //     {
// //         GameObject annotationObject = null;

// //         // Try prefab first
// //         if (annotationPrefab != null)
// //         {
// //             try
// //             {
// //                 annotationObject = Instantiate(annotationPrefab, position, Quaternion.identity);
// //                 Debug.Log($"   ✅ Created with prefab: {annotationPrefab.name}");
// //             }
// //             catch (System.Exception e)
// //             {
// //                 Debug.LogError($"   ❌ Prefab failed: {e.Message}");
// //                 annotationObject = null;
// //             }
// //         }

// //         // Create sphere if prefab failed
// //         if (annotationObject == null)
// //         {
// //             annotationObject = GameObject.CreatePrimitive(PrimitiveType.Sphere);
// //             annotationObject.transform.position = position;
// //             Debug.Log($"   🔨 Created primitive sphere");
// //         }

// //         // UNIFORM SIZE - ALL annotations same size
// //         annotationObject.transform.localScale = Vector3.one * annotationSize;

// //         // Setup
// //         placedAnnotations.Add(annotationObject);
// //         annotationObject.name = $"Annotation_{annotation.id}";

// //         // Make visible
// //         MakeVisible(annotationObject);

// //         Debug.Log($"📍 CREATED: {annotationObject.name} at {position} with size {annotationSize}");
// //     }

// //     private void MakeVisible(GameObject obj)
// //     {
// //         Renderer renderer = obj.GetComponent<Renderer>();
// //         if (renderer != null)
// //         {
// //             Material mat = new Material(Shader.Find("Standard"));
// //             mat.color = Color.red;
// //             mat.SetFloat("_Metallic", 0f);
// //             mat.SetFloat("_Glossiness", 1f);
// //             mat.EnableKeyword("_EMISSION");
// //             mat.SetColor("_EmissionColor", Color.red * 0.3f);
// //             renderer.material = mat;
// //         }
// //     }

// //     private void HandleAnnotationError(string error)
// //     {
// //         Debug.LogWarning($"⚠️ Annotation Error: {error}");
// //     }

// //     public void ClearAllAnnotations()
// //     {
// //         Debug.Log($"🧹 CLEARING {placedAnnotations.Count} annotations");

// //         for (int i = 0; i < placedAnnotations.Count; i++)
// //         {
// //             if (placedAnnotations[i] != null)
// //             {
// //                 Destroy(placedAnnotations[i]);
// //             }
// //         }

// //         placedAnnotations.Clear();
// //         processedAnnotationIds.Clear();
// //         Debug.Log($"✅ All annotations cleared");
// //     }

// //     [ContextMenu("Create Test Annotation")]
// //     public void CreateTestAnnotation()
// //     {
// //         Debug.Log($"🧪 CREATING TEST ANNOTATION");

// //         if (arCamera == null)
// //         {
// //             Debug.LogError("❌ No camera");
// //             return;
// //         }

// //         Vector2 centerScreen = new Vector2(arCamera.pixelWidth / 2f, arCamera.pixelHeight / 2f);
// //         Vector3 testPos = GetGuaranteedPosition(centerScreen);

// //         var testAnnotation = new ServerCommunication.AnnotationData
// //         {
// //             id = 999,
// //             center = new int[] { (int)centerScreen.x, (int)centerScreen.y }
// //         };

// //         CreateAnnotation(testPos, testAnnotation);
// //         Debug.Log($"🧪 Test annotation created at screen center!");
// //     }

// //     [ContextMenu("Test Coordinate (257, 286)")]
// //     public void TestProblematicCoordinate()
// //     {
// //         float testX = 257f;
// //         float testY = 286f;

// //         Debug.Log($"🧪 TESTING COORDINATE ({testX}, {testY})");

// //         if (arCamera == null)
// //         {
// //             Debug.LogError("❌ No camera");
// //             return;
// //         }

// //         Vector2 screenPoint = TransformToScreen(testX, testY);
// //         Debug.Log($"   Formula: ({testX}/640, (1-{testY}/480)) × ({arCamera.pixelWidth}, {arCamera.pixelHeight})");
// //         Debug.Log($"   Result: ({screenPoint.x:F1}, {screenPoint.y:F1})");

// //         Vector3 position = GetGuaranteedPosition(screenPoint);

// //         var testAnnotation = new ServerCommunication.AnnotationData
// //         {
// //             id = 888,
// //             center = new int[] { (int)testX, (int)testY }
// //         };

// //         CreateAnnotation(position, testAnnotation);
// //         Debug.Log($"🧪 Test annotation created for ({testX}, {testY})!");
// //     }

// //     [ContextMenu("Create Multiple Test Annotations")]
// //     public void CreateMultipleTestAnnotations()
// //     {
// //         Debug.Log($"🧪 CREATING MULTIPLE TEST ANNOTATIONS (all size {annotationSize})");

// //         if (arCamera == null)
// //         {
// //             Debug.LogError("❌ No camera");
// //             return;
// //         }

// //         Vector2[] testPositions = new Vector2[]
// //         {
// //             new Vector2(320f, 240f),  // Center
// //             new Vector2(160f, 120f),  // Top-left quadrant
// //             new Vector2(480f, 120f),  // Top-right quadrant
// //             new Vector2(160f, 360f),  // Bottom-left quadrant
// //             new Vector2(480f, 360f)   // Bottom-right quadrant
// //         };

// //         for (int i = 0; i < testPositions.Length; i++)
// //         {
// //             Vector2 annotationPos = testPositions[i];
// //             Vector2 screenPoint = TransformToScreen(annotationPos.x, annotationPos.y);
// //             Vector3 worldPos = GetGuaranteedPosition(screenPoint);

// //             var testAnnotation = new ServerCommunication.AnnotationData
// //             {
// //                 id = 900 + i,
// //                 center = new int[] { (int)annotationPos.x, (int)annotationPos.y }
// //             };

// //             CreateAnnotation(worldPos, testAnnotation);
// //         }

// //         Debug.Log($"🧪 Created {testPositions.Length} test annotations, all size {annotationSize}");
// //     }

// //     [ContextMenu("Debug System")]
// //     public void DebugSystem()
// //     {
// //         Debug.Log($"🔍 SYSTEM DEBUG:");
// //         Debug.Log($"   Camera: {(Camera.main != null ? Camera.main.name : "❌ MISSING")}");
// //         Debug.Log($"   ARRaycastManager: {(raycastManager != null ? "✅" : "❌")}");
// //         Debug.Log($"   Annotation prefab: {(annotationPrefab != null ? annotationPrefab.name : "❌ NONE")}");
// //         Debug.Log($"   Uniform size: {annotationSize}");
// //         Debug.Log($"   Current annotations: {placedAnnotations.Count}");
// //         Debug.Log($"   Guaranteed placement: ENABLED");
// //         Debug.Log($"   Fallback depth: {fallbackDepth}m");

// //         if (arCamera != null)
// //         {
// //             Debug.Log($"   Screen: {arCamera.pixelWidth}x{arCamera.pixelHeight}");
// //             Debug.Log($"   FOV: {arCamera.fieldOfView:F1}°");
// //             Debug.Log($"   Scale factors: X={arCamera.pixelWidth/640f:F3}, Y={arCamera.pixelHeight/480f:F3}");
// //         }

// //         // List annotations
// //         Debug.Log($"   📋 ANNOTATIONS ({placedAnnotations.Count}):");
// //         for (int i = 0; i < placedAnnotations.Count; i++)
// //         {
// //             if (placedAnnotations[i] != null)
// //             {
// //                 Vector3 pos = placedAnnotations[i].transform.position;
// //                 Vector3 scale = placedAnnotations[i].transform.localScale;
// //                 float distance = arCamera != null ? Vector3.Distance(pos, arCamera.transform.position) : 0f;

// //                 Debug.Log($"      {i + 1}. {placedAnnotations[i].name} at {pos} (scale: {scale.x:F3}, dist: {distance:F1}m)");
// //             }
// //         }
// //     }
// // }

// using System;
// using System.Collections;
// using System.Collections.Generic;
// using UnityEngine;
// using UnityEngine.XR.ARFoundation;

// public class ARPlaceCube : MonoBehaviour
// {
//     [Header("AR Components")]
//     [SerializeField] private ARRaycastManager raycastManager;
//     [SerializeField] private ARPlaneManager planeManager;
//     [SerializeField] private ARPointCloudManager pointCloudManager;

//     [Header("Server Communication")]
//     [SerializeField] private ServerCommunication serverCommunication;

//     [Header("Annotation Settings")]
//     [SerializeField] private GameObject annotationPrefab;
//     [SerializeField] private bool enableAnnotationSystem = true;
//     [SerializeField] private float annotationSize = 0.06f; // UNIFIED SIZE FOR ALL ANNOTATIONS

//     [Header("Auto-Clear Settings")]
//     [SerializeField] private bool clearPreviousAnnotations = true; // NEW: Enable/disable auto-clearing
//     [SerializeField] private bool clearOnNewAnnotation = true;     // NEW: Clear when receiving new server annotation
//     [SerializeField] private bool clearOnTestAnnotation = true;    // NEW: Clear when creating test annotations

//     [Header("Coordinate Calibration - FIND THE RIGHT VALUES")]
//     [SerializeField] private float xOffset = 0.15f;        // Adjustable X offset for fine-tuning
//     [SerializeField] private float yMultiplier = 1.2f;     // Adjustable Y multiplier for fine-tuning
//     [SerializeField] private bool useCustomTransform = true; // Toggle between standard and custom transform
    
//     [Header("Sub-Pixel Fine-Tuning")]
//     [SerializeField] private float microXOffset = 0.1f;    // Tiny X adjustment for final precision
//     [SerializeField] private float microYOffset = 0.1f;    // Tiny Y adjustment for final precision
//     [SerializeField] private bool enableSubPixelTuning = true; // Enable micro-adjustments

//     [Header("Guaranteed Placement - ALWAYS CREATE ANNOTATIONS")]
//     [SerializeField] private bool alwaysCreateAnnotation = true; 
//     [SerializeField] private float fallbackDepth = 2.0f;
//     [SerializeField] private bool forceGuaranteedPlacement = true;
//     [SerializeField] private bool preferHitTestWhenAvailable = true;

//     [Header("Debug")]
//     [SerializeField] private bool debugAnnotations = true;
//     [SerializeField] private bool showDetailedLogs = true;
//     [SerializeField] private bool createTestAnnotationOnStart = false;

//     private List<GameObject> placedAnnotations = new List<GameObject>();
//     private List<int> processedAnnotationIds = new List<int>();
//     private Camera arCamera;

//     void Start()
//     {
//         if (enableAnnotationSystem && serverCommunication != null)
//         {
//             InitializeAnnotationSystem();
//         }

//         InitializeARComponents();
//         arCamera = Camera.main ?? FindObjectOfType<Camera>();

//         Debug.Log("🎯 SIMPLE ANNOTATION PLACEMENT SYSTEM - NO COMPLEX TRANSFORMS");
//         LogSystemStatus();
//         CheckPrefabAssignment();

//         if (createTestAnnotationOnStart)
//         {
//             Debug.Log("🧪 Creating test annotation on start");
//             Invoke(nameof(CreateTestAnnotation), 2f);
//         }
//     }

//     void OnDestroy()
//     {
//         if (enableAnnotationSystem && serverCommunication != null)
//         {
//             serverCommunication.OnNewAnnotationReceived -= HandleNewAnnotation;
//             serverCommunication.OnAnnotationError -= HandleAnnotationError;
//         }
//     }

//     private void InitializeARComponents()
//     {
//         if (raycastManager == null)
//             raycastManager = FindObjectOfType<ARRaycastManager>();

//         if (planeManager == null)
//             planeManager = FindObjectOfType<ARPlaneManager>();

//         if (pointCloudManager == null)
//             pointCloudManager = FindObjectOfType<ARPointCloudManager>();

//         Debug.Log($"🔧 AR COMPONENTS:");
//         Debug.Log($"   Raycast Manager: {(raycastManager != null ? "✅" : "❌")}");
//         Debug.Log($"   Plane Manager: {(planeManager != null ? "✅" : "❌")}");
//         Debug.Log($"   Point Cloud Manager: {(pointCloudManager != null ? "✅" : "❌")}");
//     }

//     private void LogSystemStatus()
//     {
//         Debug.Log($"   📏 UNIFORM SIZE: {annotationSize}");
//         Debug.Log($"   🎯 GUARANTEED PLACEMENT: ENABLED");
//         Debug.Log($"   🧹 AUTO-CLEAR PREVIOUS: {(clearPreviousAnnotations ? "ENABLED" : "DISABLED")}");
//         Debug.Log($"   📐 Fallback depth: {fallbackDepth}m");
//         Debug.Log($"   🖼️ SIMPLE FORMULA: (x/640, (1-y/480)) × screen");
//         Debug.Log($"   📱 Screen: {(arCamera ? $"{arCamera.pixelWidth}x{arCamera.pixelHeight}" : "Unknown")}");

//         if (arCamera != null)
//         {
//             float scaleX = arCamera.pixelWidth / 640f;
//             float scaleY = arCamera.pixelHeight / 480f;
//             Debug.Log($"   📐 Scale: X={scaleX:F3}, Y={scaleY:F3}");
//         }
//     }

//     private void CheckPrefabAssignment()
//     {
//         if (annotationPrefab == null)
//         {
//             Debug.LogWarning("⚠️ No prefab - will create spheres");
//         }
//         else
//         {
//             Debug.Log($"✅ Prefab: {annotationPrefab.name}");
//         }
//     }

//     private void InitializeAnnotationSystem()
//     {
//         serverCommunication.OnNewAnnotationReceived += HandleNewAnnotation;
//         serverCommunication.OnAnnotationError += HandleAnnotationError;
//         serverCommunication.StartAnnotationPolling(2.0f);
//         Debug.Log("🎯 Annotation system initialized");
//     }

//     /// <summary>
//     /// NEW: Clear previous annotations before creating new ones
//     /// </summary>
//     private void ClearPreviousAnnotationsIfEnabled(string context = "")
//     {
//         if (!clearPreviousAnnotations)
//         {
//             if (showDetailedLogs)
//                 Debug.Log($"   🔒 Auto-clear disabled - keeping {placedAnnotations.Count} existing annotations");
//             return;
//         }

//         if (placedAnnotations.Count > 0)
//         {
//             Debug.Log($"🧹 Clearing {placedAnnotations.Count} previous annotations{(string.IsNullOrEmpty(context) ? "" : $" ({context})")}");
//             ClearAllAnnotations();
//         }
//         else
//         {
//             if (showDetailedLogs)
//                 Debug.Log($"   ℹ️ No previous annotations to clear{(string.IsNullOrEmpty(context) ? "" : $" ({context})")}");
//         }
//     }

//     private void HandleNewAnnotation(ServerCommunication.AnnotationData annotation)
//     {
//         Debug.Log("═══════════════════════════════════════");
//         Debug.Log($"🎯 ANNOTATION: ID {annotation.id}");

//         // NEW: Clear previous annotations if enabled
//         if (clearOnNewAnnotation)
//         {
//             ClearPreviousAnnotationsIfEnabled("new server annotation");
//         }

//         float annotationX = annotation.center[0];
//         float annotationY = annotation.center[1];

//         Debug.Log($"📍 Input: ({annotationX}, {annotationY}) in 640x480");

//         // SIMPLE COORDINATE TRANSFORMATION
//         Vector2 screenPoint = TransformToScreen(annotationX, annotationY);
//         Debug.Log($"📍 Screen: ({screenPoint.x:F1}, {screenPoint.y:F1})");

//         // GUARANTEED POSITION
//         Vector3 position = GetGuaranteedPosition(screenPoint);

//         // CREATE ANNOTATION
//         CreateAnnotation(position, annotation);

//         processedAnnotationIds.Add(annotation.id);
//         serverCommunication.MarkAnnotationsAsProcessed(new int[] { annotation.id });

//         Debug.Log($"✅ ANNOTATION PLACED at {position}");
//         Debug.Log($"📍 Total annotations: {placedAnnotations.Count}");
//         Debug.Log("═══════════════════════════════════════");
//     }

//     /// <summary>
//     /// ULTRA-PRECISE: Transform with sub-pixel fine-tuning for perfect alignment
//     /// </summary>
//     private Vector2 TransformToScreen(float annotationX, float annotationY)
//     {
//         if (arCamera == null)
//         {
//             return new Vector2(annotationX, annotationY);
//         }

//         // Step 1: Normalize to 0-1 range
//         float normalizedX = annotationX / 640f;
//         float normalizedY = annotationY / 480f;

//         if (useCustomTransform)
//         {
//             // YOUR WORKING APPROACH - with adjustable parameters
//             normalizedX = normalizedX + xOffset;
//             normalizedY = yMultiplier - normalizedY;
            
//             // SUB-PIXEL FINE-TUNING for minimal misalignment
//             if (enableSubPixelTuning)
//             {
//                 normalizedX += microXOffset;
//                 normalizedY += microYOffset;
//             }
            
//             if (showDetailedLogs)
//             {
//                 Debug.Log($"   🔧 CUSTOM Transform: X+{xOffset}+{microXOffset}, Y={yMultiplier}-Y+{microYOffset}");
//             }
//         }
//         else
//         {
//             // STANDARD APPROACH - for comparison
//             normalizedY = 1.0f - normalizedY;
            
//             if (showDetailedLogs)
//             {
//                 Debug.Log($"   📐 STANDARD Transform: Y=1-Y");
//             }
//         }

//         // Step 3: Scale to actual screen resolution
//         float screenX = normalizedX * arCamera.pixelWidth;
//         float screenY = normalizedY * arCamera.pixelHeight;

//         // Step 4: Clamp to valid screen bounds (but allow slight overflow for precision)
//         screenX = Mathf.Clamp(screenX, -5f, arCamera.pixelWidth + 5f);
//         screenY = Mathf.Clamp(screenY, -5f, arCamera.pixelHeight + 5f);

//         if (showDetailedLogs)
//         {
//             Debug.Log($"   📍 Ultra-Precise Transform:");
//             Debug.Log($"      Input: ({annotationX}, {annotationY}) in 640x480");
//             Debug.Log($"      Normalized: ({normalizedX:F6}, {normalizedY:F6})");
//             Debug.Log($"      Screen: ({screenX:F2}, {screenY:F2}) in {arCamera.pixelWidth}x{arCamera.pixelHeight}");
//         }

//         return new Vector2(screenX, screenY);
//     }

//     /// <summary>
//     /// GUARANTEED: Always returns a valid position
//     /// </summary>
//     private Vector3 GetGuaranteedPosition(Vector2 screenPoint)
//     {
//         Debug.Log($"🔍 Getting position for screen ({screenPoint.x:F1}, {screenPoint.y:F1})");

//         // TRY 1: AR Raycast
//         if (preferHitTestWhenAvailable && raycastManager != null)
//         {
//             Vector3 arHit = TryARRaycast(screenPoint);
//             if (arHit != Vector3.zero)
//             {
//                 Debug.Log($"✅ AR Hit: {arHit}");
//                 return arHit;
//             }
//         }

//         // TRY 2: Physics Raycast
//         if (preferHitTestWhenAvailable)
//         {
//             Vector3 physicsHit = TryPhysicsRaycast(screenPoint);
//             if (physicsHit != Vector3.zero)
//             {
//                 Debug.Log($"✅ Physics Hit: {physicsHit}");
//                 return physicsHit;
//             }
//         }

//         // GUARANTEED: Camera ray at fixed depth
//         Vector3 guaranteedPos = GetCameraRayPosition(screenPoint);
//         Debug.Log($"✅ Guaranteed: {guaranteedPos}");
//         return guaranteedPos;
//     }

//     /// <summary>
//     /// Try AR raycast - return zero if no hit
//     /// </summary>
//     private Vector3 TryARRaycast(Vector2 screenPoint)
//     {
//         if (raycastManager == null) return Vector3.zero;

//         var hits = new List<ARRaycastHit>();
        
//         if (raycastManager.Raycast(screenPoint, hits, UnityEngine.XR.ARSubsystems.TrackableType.AllTypes))
//         {
//             float distance = Vector3.Distance(arCamera.transform.position, hits[0].pose.position);
//             Debug.Log($"   AR raycast HIT at {distance:F2}m");
//             return hits[0].pose.position;
//         }

//         Debug.Log($"   AR raycast MISS");
//         return Vector3.zero;
//     }

//     /// <summary>
//     /// Try physics raycast - return zero if no hit
//     /// </summary>
//     private Vector3 TryPhysicsRaycast(Vector2 screenPoint)
//     {
//         if (arCamera == null) return Vector3.zero;

//         Ray ray = arCamera.ScreenPointToRay(new Vector3(screenPoint.x, screenPoint.y, 0));
//         RaycastHit hit;

//         if (Physics.Raycast(ray, out hit, 20f))
//         {
//             Debug.Log($"   Physics raycast HIT {hit.collider.name} at {hit.distance:F2}m");
//             return hit.point;
//         }

//         Debug.Log($"   Physics raycast MISS");
//         return Vector3.zero;
//     }

//     /// <summary>
//     /// GUARANTEED: Use camera ray at fixed depth - ALWAYS works
//     /// </summary>
//     private Vector3 GetCameraRayPosition(Vector2 screenPoint)
//     {
//         if (arCamera == null)
//         {
//             // No camera - place in world forward
//             return Vector3.forward * fallbackDepth;
//         }

//         // Use Unity's screen-to-ray for perfect coordinate mapping
//         Ray ray = arCamera.ScreenPointToRay(new Vector3(screenPoint.x, screenPoint.y, 0));
//         Vector3 position = ray.origin + ray.direction.normalized * fallbackDepth;

//         if (showDetailedLogs)
//         {
//             Debug.Log($"   Camera ray: origin={ray.origin}, dir={ray.direction.normalized}, pos={position}");
//         }

//         return position;
//     }

//     /// <summary>
//     /// Create annotation with SIMPLE VISIBLE CIRCLE that works reliably in AR
//     /// </summary>
//     private void CreateAnnotation(Vector3 position, ServerCommunication.AnnotationData annotation)
//     {
//         GameObject annotationObject = CreateVisibleCircle(position);

//         // UNIFORM SIZE - ALL annotations same size
//         annotationObject.transform.localScale = Vector3.one * annotationSize;

//         // Setup
//         placedAnnotations.Add(annotationObject);
//         annotationObject.name = $"Annotation_{annotation.id}";

//         Debug.Log($"📍 CREATED: {annotationObject.name} at {position} with size {annotationSize}");
        
//         // VERIFY it's actually visible
//         VerifyAnnotationVisibility(annotationObject);
//     }

//     /// <summary>
//     /// Create a SIMPLE but VISIBLE circle using a quad with custom material - GUARANTEED to work in AR
//     /// </summary>
//     private GameObject CreateVisibleCircle(Vector3 position)
//     {
//         // Create a simple quad (flat rectangle)
//         GameObject circleObject = GameObject.CreatePrimitive(PrimitiveType.Quad);
//         circleObject.transform.position = position;
//         circleObject.name = "CircleAnnotation";

//         // Remove the collider (we don't need it)
//         Collider quadCollider = circleObject.GetComponent<Collider>();
//         if (quadCollider != null)
//             Destroy(quadCollider);

//         // Get the renderer
//         Renderer renderer = circleObject.GetComponent<Renderer>();
//         if (renderer != null)
//         {
//             // Create circle texture
//             Texture2D circleTexture = CreateCircleTexture();
            
//             // Create material with the circle texture
//             Material circleMaterial = new Material(Shader.Find("Sprites/Default"));
//             circleMaterial.mainTexture = circleTexture;
            
//             // Make it transparent but VISIBLE
//             circleMaterial.color = new Color(1f, 0f, 0f, 0.5f); // 60% transparent red
            
//             // Enable transparency
//             circleMaterial.SetFloat("_Mode", 3);
//             circleMaterial.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
//             circleMaterial.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
//             circleMaterial.SetInt("_ZWrite", 0);
//             circleMaterial.EnableKeyword("_ALPHABLEND_ON");
//             circleMaterial.renderQueue = 3000;
            
//             renderer.material = circleMaterial;
            
//             Debug.Log($"   🔴 Applied circle texture with transparency");
//         }

//         // Make it always face the camera
//         circleObject.AddComponent<SimpleBillboard>();

//         Debug.Log($"   ✅ Created SIMPLE visible circle at {position}");
//         return circleObject;
//     }

//     /// <summary>
//     /// Create circle texture with transparent center and red outline
//     /// </summary>
//     private Texture2D CreateCircleTexture()
//     {
//         int size = 128;
//         Texture2D texture = new Texture2D(size, size, TextureFormat.RGBA32, false);
//         Color[] pixels = new Color[size * size];

//         Vector2 center = new Vector2(size / 2f, size / 2f);
//         float outerRadius = size / 2f - 2f;
//         float innerRadius = outerRadius - 4f; // Border thickness

//         for (int y = 0; y < size; y++)
//         {
//             for (int x = 0; x < size; x++)
//             {
//                 Vector2 pos = new Vector2(x, y);
//                 float distance = Vector2.Distance(pos, center);
                
//                 if (distance <= innerRadius)
//                 {
//                     // Inside circle - transparent
//                     pixels[y * size + x] = new Color(1f, 0f, 0f, 0.1f); // Very transparent red
//                 }
//                 else if (distance <= outerRadius)
//                 {
//                     // Border - solid red
//                     pixels[y * size + x] = new Color(1f, 0f, 0f, 1f); // Solid red
//                 }
//                 else
//                 {
//                     // Outside circle - completely transparent
//                     pixels[y * size + x] = Color.clear;
//                 }
//             }
//         }

//         texture.SetPixels(pixels);
//         texture.Apply();
//         texture.filterMode = FilterMode.Bilinear;

//         Debug.Log($"   🎨 Created circle texture {size}x{size}");
//         return texture;
//     }

//     /// <summary>
//     /// Simple billboard that always faces camera - more reliable than complex Canvas setup
//     /// </summary>
//     public class SimpleBillboard : MonoBehaviour
//     {
//         private Camera targetCamera;

//         void Start()
//         {
//             targetCamera = Camera.main ?? FindObjectOfType<Camera>();
//             if (targetCamera == null)
//             {
//                 Debug.LogWarning("⚠️ No camera found for billboard");
//             }
//         }

//         void Update()
//         {
//             if (targetCamera != null)
//             {
//                 // Always look at camera
//                 Vector3 lookDirection = targetCamera.transform.position - transform.position;
//                 transform.rotation = Quaternion.LookRotation(-lookDirection);
//             }
//         }
//     }

//     /// <summary>
//     /// Verify that the annotation is actually visible and properly positioned
//     /// </summary>
//     private void VerifyAnnotationVisibility(GameObject annotation)
//     {
//         if (annotation == null)
//         {
//             Debug.LogError("❌ Annotation is null!");
//             return;
//         }

//         Renderer renderer = annotation.GetComponent<Renderer>();
//         if (renderer == null)
//         {
//             Debug.LogError($"❌ {annotation.name} has no renderer!");
//             return;
//         }

//         if (!renderer.enabled)
//         {
//             Debug.LogWarning($"⚠️ {annotation.name} renderer is disabled!");
//             renderer.enabled = true;
//         }

//         if (renderer.material == null)
//         {
//             Debug.LogError($"❌ {annotation.name} has no material!");
//             return;
//         }

//         // Check if it's in camera view
//         if (arCamera != null)
//         {
//             Vector3 screenPos = arCamera.WorldToScreenPoint(annotation.transform.position);
//             bool inView = screenPos.z > 0 && screenPos.x >= 0 && screenPos.x <= arCamera.pixelWidth 
//                          && screenPos.y >= 0 && screenPos.y <= arCamera.pixelHeight;
            
//             float distance = Vector3.Distance(arCamera.transform.position, annotation.transform.position);
            
//             Debug.Log($"   👁️ VISIBILITY CHECK for {annotation.name}:");
//             Debug.Log($"      Position: {annotation.transform.position}");
//             Debug.Log($"      Screen pos: ({screenPos.x:F1}, {screenPos.y:F1}, {screenPos.z:F2})");
//             Debug.Log($"      In camera view: {(inView ? "✅ YES" : "❌ NO")}");
//             Debug.Log($"      Distance: {distance:F2}m");
//             Debug.Log($"      Scale: {annotation.transform.localScale}");
//             Debug.Log($"      Renderer enabled: {renderer.enabled}");
//             Debug.Log($"      Material: {renderer.material.name}");
//         }
//     }

//     private void HandleAnnotationError(string error)
//     {
//         Debug.LogWarning($"⚠️ Annotation Error: {error}");
//     }

//     public void ClearAllAnnotations()
//     {
//         Debug.Log($"🧹 CLEARING {placedAnnotations.Count} annotations");
        
//         for (int i = 0; i < placedAnnotations.Count; i++)
//         {
//             if (placedAnnotations[i] != null)
//             {
//                 Destroy(placedAnnotations[i]);
//             }
//         }

//         placedAnnotations.Clear();
//         processedAnnotationIds.Clear();
//         Debug.Log($"✅ All annotations cleared");
//     }

//     [ContextMenu("Create Test Annotation")]
//     public void CreateTestAnnotation()
//     {
//         Debug.Log($"🧪 CREATING TEST ANNOTATION");

//         if (arCamera == null)
//         {
//             Debug.LogError("❌ No camera");
//             return;
//         }

//         // NEW: Clear previous annotations if enabled for test annotations
//         if (clearOnTestAnnotation)
//         {
//             ClearPreviousAnnotationsIfEnabled("test annotation");
//         }

//         Vector2 centerScreen = new Vector2(arCamera.pixelWidth / 2f, arCamera.pixelHeight / 2f);
//         Vector3 testPos = GetGuaranteedPosition(centerScreen);

//         var testAnnotation = new ServerCommunication.AnnotationData
//         {
//             id = 999,
//             center = new int[] { (int)centerScreen.x, (int)centerScreen.y }
//         };

//         CreateAnnotation(testPos, testAnnotation);
//         Debug.Log($"🧪 Test annotation created at screen center!");
//     }

//     [ContextMenu("Ultra-Fine Positioning")]
//     public void UltraFineTuning()
//     {
//         Debug.Log($"🔬 ULTRA-FINE TUNING with micro-adjustments");
//         Debug.Log($"   Main: X+{xOffset}, Y={yMultiplier}-Y");
//         Debug.Log($"   Micro: X+{microXOffset:F4}, Y+{microYOffset:F4}");
        
//         if (arCamera == null)
//         {
//             Debug.LogError("❌ No camera");
//             return;
//         }

//         if (clearOnTestAnnotation)
//         {
//             ClearPreviousAnnotationsIfEnabled("ultra-fine test");
//         }

//         // Test center point with current settings
//         float centerX = 320f;
//         float centerY = 240f;
        
//         Vector2 screenPoint = TransformToScreen(centerX, centerY);
//         Vector2 expectedCenter = new Vector2(arCamera.pixelWidth / 2f, arCamera.pixelHeight / 2f);
        
//         float pixelErrorX = screenPoint.x - expectedCenter.x;
//         float pixelErrorY = screenPoint.y - expectedCenter.y;
        
//         // Calculate sub-pixel corrections
//         float suggestedMicroX = microXOffset - (pixelErrorX / arCamera.pixelWidth);
//         float suggestedMicroY = microYOffset - (pixelErrorY / arCamera.pixelHeight);
        
//         Debug.Log($"📍 Current position: ({screenPoint.x:F2}, {screenPoint.y:F2})");
//         Debug.Log($"📍 Expected center: ({expectedCenter.x:F2}, {expectedCenter.y:F2})");
//         Debug.Log($"📏 Pixel error: X={pixelErrorX:F2}, Y={pixelErrorY:F2}");
//         Debug.Log($"💡 Try micro adjustments:");
//         Debug.Log($"   microXOffset: {suggestedMicroX:F6} (current: {microXOffset:F6})");
//         Debug.Log($"   microYOffset: {suggestedMicroY:F6} (current: {microYOffset:F6})");
        
//         Vector3 position = GetGuaranteedPosition(screenPoint);
        
//         var testAnnotation = new ServerCommunication.AnnotationData
//         {
//             id = 777,
//             center = new int[] { (int)centerX, (int)centerY }
//         };
        
//         CreateAnnotation(position, testAnnotation);
//         Debug.Log($"🔬 Ultra-fine test annotation created!");
//     }

//     [ContextMenu("Test Known Coordinate")]
//     public void TestKnownCoordinate()
//     {
//         // Use the coordinate you mentioned: (257, 286)
//         float testX = 257f;
//         float testY = 286f;

//         Debug.Log($"🧪 TESTING KNOWN COORDINATE ({testX}, {testY})");
//         Debug.Log($"🔧 Using: X offset = {xOffset}, Y multiplier = {yMultiplier}");

//         if (arCamera == null)
//         {
//             Debug.LogError("❌ No camera");
//             return;
//         }

//         if (clearOnTestAnnotation)
//         {
//             ClearPreviousAnnotationsIfEnabled("known coordinate test");
//         }

//         Vector2 screenPoint = TransformToScreen(testX, testY);
//         Debug.Log($"   Formula: (({testX}/640) + {xOffset}, {yMultiplier} - ({testY}/480)) × screen");
//         Debug.Log($"   Result: ({screenPoint.x:F1}, {screenPoint.y:F1})");

//         Vector3 position = GetGuaranteedPosition(screenPoint);

//         var testAnnotation = new ServerCommunication.AnnotationData
//         {
//             id = 888,
//             center = new int[] { (int)testX, (int)testY }
//         };

//         CreateAnnotation(position, testAnnotation);
//         Debug.Log($"🧪 Known coordinate test created for ({testX}, {testY})!");
//     }

//     [ContextMenu("Test All Corners")]
//     public void TestAllCorners()
//     {
//         Debug.Log($"🎯 TESTING ALL CORNERS - Should be at screen edges");
        
//         if (arCamera == null)
//         {
//             Debug.LogError("❌ No camera");
//             return;
//         }

//         if (clearOnTestAnnotation)
//         {
//             ClearPreviousAnnotationsIfEnabled("corner test");
//         }

//         // Test corners in 640x480 space
//         Vector2[] corners = new Vector2[]
//         {
//             new Vector2(0f, 0f),       // Top-left
//             new Vector2(640f, 0f),     // Top-right  
//             new Vector2(0f, 480f),     // Bottom-left
//             new Vector2(640f, 480f),   // Bottom-right
//             new Vector2(320f, 240f)    // Center
//         };

//         string[] names = { "Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right", "Center" };

//         for (int i = 0; i < corners.Length; i++)
//         {
//             Vector2 corner = corners[i];
//             Vector2 screenPoint = TransformToScreen(corner.x, corner.y);
//             Vector3 position = GetGuaranteedPosition(screenPoint);
            
//             var testAnnotation = new ServerCommunication.AnnotationData
//             {
//                 id = 700 + i,
//                 center = new int[] { (int)corner.x, (int)corner.y }
//             };
            
//             CreateAnnotation(position, testAnnotation);
//             Debug.Log($"📍 {names[i]}: ({corner.x}, {corner.y}) → screen({screenPoint.x:F1}, {screenPoint.y:F1})");
//         }
        
//         Debug.Log($"🎯 Created {corners.Length} corner test annotations");
//     }

//     /// <summary>
//     /// Advanced calibration function with custom offsets
//     /// </summary>
//     [ContextMenu("Calibrate Positioning")]
//     public void CalibratePositioning()
//     {
//         Debug.Log($"🔧 CALIBRATION MODE - Testing different coordinate mappings");
        
//         if (arCamera == null)
//         {
//             Debug.LogError("❌ No camera");
//             return;
//         }

//         Debug.Log($"📱 Screen Resolution: {arCamera.pixelWidth}x{arCamera.pixelHeight}");
//         Debug.Log($"📐 Aspect Ratio: {(float)arCamera.pixelWidth / arCamera.pixelHeight:F2}");
//         Debug.Log($"📏 Scale Factors: X={arCamera.pixelWidth/640f:F3}, Y={arCamera.pixelHeight/480f:F3}");
        
//         // Calculate if there might be letterboxing/pillarboxing
//         float sourceAspect = 640f / 480f; // 1.333
//         float screenAspect = (float)arCamera.pixelWidth / arCamera.pixelHeight;
        
//         Debug.Log($"🎭 Source aspect: {sourceAspect:F3}, Screen aspect: {screenAspect:F3}");
        
//         if (Mathf.Abs(sourceAspect - screenAspect) > 0.1f)
//         {
//             Debug.LogWarning($"⚠️ ASPECT RATIO MISMATCH - This might cause positioning issues!");
//             Debug.LogWarning($"   Consider aspect ratio correction in coordinate transformation");
//         }
        
//         // Test the exact center
//         TestCenterPoint();
//     }

//     private void TestCenterPoint()
//     {
//         throw new NotImplementedException();
//     }

//     [ContextMenu("Create Multiple Test Annotations")]
//     public void CreateMultipleTestAnnotations()
//     {
//         Debug.Log($"🧪 CREATING MULTIPLE TEST ANNOTATIONS (all size {annotationSize})");

//         if (arCamera == null)
//         {
//             Debug.LogError("❌ No camera");
//             return;
//         }

//         // NEW: Clear previous annotations if enabled for test annotations
//         if (clearOnTestAnnotation)
//         {
//             ClearPreviousAnnotationsIfEnabled("multiple test annotations");
//         }

//         Vector2[] testPositions = new Vector2[]
//         {
//             new Vector2(320f, 240f),  // Center
//             new Vector2(160f, 120f),  // Top-left quadrant
//             new Vector2(480f, 120f),  // Top-right quadrant
//             new Vector2(160f, 360f),  // Bottom-left quadrant
//             new Vector2(480f, 360f)   // Bottom-right quadrant
//         };

//         for (int i = 0; i < testPositions.Length; i++)
//         {
//             Vector2 annotationPos = testPositions[i];
//             Vector2 screenPoint = TransformToScreen(annotationPos.x, annotationPos.y);
//             Vector3 worldPos = GetGuaranteedPosition(screenPoint);

//             var testAnnotation = new ServerCommunication.AnnotationData
//             {
//                 id = 900 + i,
//                 center = new int[] { (int)annotationPos.x, (int)annotationPos.y }
//             };

//             CreateAnnotation(worldPos, testAnnotation);
//         }

//         Debug.Log($"🧪 Created {testPositions.Length} test annotations, all size {annotationSize}");
//     }

//     /// <summary>
//     /// NEW: Toggle the auto-clear functionality at runtime
//     /// </summary>
//     [ContextMenu("Toggle Auto-Clear")]
//     public void ToggleAutoClear()
//     {
//         clearPreviousAnnotations = !clearPreviousAnnotations;
//         Debug.Log($"🔄 Auto-clear previous annotations: {(clearPreviousAnnotations ? "ENABLED" : "DISABLED")}");
//     }

//     /// <summary>
//     /// NEW: Create annotation with manual clear control
//     /// </summary>
//     public void CreateAnnotationWithClear(Vector3 position, ServerCommunication.AnnotationData annotation, bool clearPrevious = true)
//     {
//         if (clearPrevious)
//         {
//             ClearPreviousAnnotationsIfEnabled("manual annotation creation");
//         }

//         CreateAnnotation(position, annotation);
//     }

//     [ContextMenu("Debug System")]
//     public void DebugSystem()
//     {
//         Debug.Log($"🔍 SYSTEM DEBUG:");
//         Debug.Log($"   Camera: {(Camera.main != null ? Camera.main.name : "❌ MISSING")}");
//         Debug.Log($"   ARRaycastManager: {(raycastManager != null ? "✅" : "❌")}");
//         Debug.Log($"   Annotation prefab: {(annotationPrefab != null ? annotationPrefab.name : "❌ NONE")}");
//         Debug.Log($"   Uniform size: {annotationSize}");
//         Debug.Log($"   Current annotations: {placedAnnotations.Count}");
//         Debug.Log($"   Guaranteed placement: ENABLED");
//         Debug.Log($"   Auto-clear previous: {(clearPreviousAnnotations ? "ENABLED" : "DISABLED")}");
//         Debug.Log($"   Clear on new annotation: {(clearOnNewAnnotation ? "ENABLED" : "DISABLED")}");
//         Debug.Log($"   Clear on test annotation: {(clearOnTestAnnotation ? "ENABLED" : "DISABLED")}");
//         Debug.Log($"   Fallback depth: {fallbackDepth}m");

//         if (arCamera != null)
//         {
//             Debug.Log($"   Screen: {arCamera.pixelWidth}x{arCamera.pixelHeight}");
//             Debug.Log($"   FOV: {arCamera.fieldOfView:F1}°");
//             Debug.Log($"   Scale factors: X={arCamera.pixelWidth/640f:F3}, Y={arCamera.pixelHeight/480f:F3}");
//         }

//         // List annotations
//         Debug.Log($"   📋 ANNOTATIONS ({placedAnnotations.Count}):");
//         for (int i = 0; i < placedAnnotations.Count; i++)
//         {
//             if (placedAnnotations[i] != null)
//             {
//                 Vector3 pos = placedAnnotations[i].transform.position;
//                 Vector3 scale = placedAnnotations[i].transform.localScale;
//                 float distance = arCamera != null ? Vector3.Distance(pos, arCamera.transform.position) : 0f;
                
//                 Debug.Log($"      {i + 1}. {placedAnnotations[i].name} at {pos} (scale: {scale.x:F3}, dist: {distance:F1}m)");
//             }
//         }
//     }
// }