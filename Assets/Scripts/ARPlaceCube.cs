using System.Collections;
using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.XR.ARFoundation;
using static UnityEngine.InputSystem.InputControlExtensions;

public class ARPlaceCube : MonoBehaviour
{
    [SerializeField] private ARRaycastManager raycastManager;
    bool isPlacing = false;

    // Update is called once per frame
    void Update()
    {
        if (!raycastManager) return;

        if ((Input.touchCount > 0 && Input.GetTouch(index: 0).phase == TouchPhase.Began) ||
                Input.GetMouseButtonDown(0) && !isPlacing)
        {
            isPlacing = true;

            if (Input.touchCount > 0)
            {
                PlaceObject(Input.GetTouch(index: 0).position);
            }
            else
            {
                PlaceObject(Input.mousePosition);
            }
        }
    }

    void PlaceObject(Vector2 touchPosition)
    {
        var rayHits = new List<ARRaycastHit>(); // Correctly use System.Collections.Generic.List
        raycastManager.Raycast(touchPosition, rayHits, UnityEngine.XR.ARSubsystems.TrackableType.AllTypes);

        if (rayHits.Count > 0)
        {
            Vector3 hitPosePosition = rayHits[0].pose.position;
            Quaternion hitPoseRotation = rayHits[0].pose.rotation;
            Instantiate(raycastManager.raycastPrefab, hitPosePosition, hitPoseRotation);
        }
        StartCoroutine(routine: SetIsPlacingToFalseWithDelay());

        if (rayHits.Count > 0)
        {
            Pose hitPose = rayHits[0].pose;
            Debug.Log($"Hit position: {hitPose.position}, rotation: {hitPose.rotation}");
        }

    }

    IEnumerator SetIsPlacingToFalseWithDelay()
    {
        yield return new WaitForSeconds(0.25f);
        isPlacing = false;
    }
}