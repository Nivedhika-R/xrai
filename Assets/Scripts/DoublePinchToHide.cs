using MixedReality.Toolkit;
using MixedReality.Toolkit.Subsystems;
using UnityEngine;
using UnityEngine.XR;

public class DoublePinchToHide : MonoBehaviour
{
    [SerializeField]
    private float doublePinchMaxTime = 0.5f; // time allowed between pinches

    [SerializeField]
    private GameObject objectToHide;

    private float lastPinchTime = -1f;
    private bool pinchInProgress = false;

    private HandsAggregatorSubsystem handsAggregator;

    void Start()
    {
        handsAggregator = XRSubsystemHelpers.GetFirstRunningSubsystem<HandsAggregatorSubsystem>();
        if (handsAggregator == null)
        {
            Debug.LogError("HandsAggregatorSubsystem not found. Ensure MRTK3 is properly configured.");
        }
    }

    void Update()
    {
        if (handsAggregator == null)
            return;

        if (handsAggregator.TryGetPinchProgress(XRNode.RightHand, out _, out bool rightIsPinching, out _))
        {
            if (rightIsPinching)
            {
                if (!pinchInProgress)
                {
                    pinchInProgress = true;

                    float currentTime = Time.time;

                    if (lastPinchTime > 0f && currentTime - lastPinchTime <= doublePinchMaxTime)
                    {
                        OnDoublePinchDetected();
                        lastPinchTime = -1f; // reset
                    }
                    else
                    {
                        lastPinchTime = currentTime;
                    }
                }
            }
            else
            {
                pinchInProgress = false;
            }
        }
    }

    private void OnDoublePinchDetected()
    {
        if (objectToHide != null)
        {
            objectToHide.SetActive(!objectToHide.activeSelf);
        }
        else
        {
            Debug.LogWarning("Object to hide is not assigned.");
        }
    }
}
