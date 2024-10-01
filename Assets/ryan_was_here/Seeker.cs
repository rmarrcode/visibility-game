using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.SideChannels;
using System;

public class Seeker : Agent
{
    private Rigidbody agentRigidbody;
    public Hider otherAgent; 
    public GameObject plane;
    private float moveCooldown = 0.2f;
    private float nextMoveTime = 0f;
    public float viewAngle = 90f;
    public LayerMask obstacleMask;
    public LayerMask agentMask;
    public float viewDistance = 100f;
    private DebugSideChannel debugSideChannel;
    public int timeStep;
    public BreadCrumbs breadCrumbs;
    public StepTrace stepTrace;

    void Start()
    {
        stepTrace = new StepTrace();
        obstacleMask = LayerMask.GetMask("Obstacle");
        agentMask = LayerMask.GetMask("Agent");
    }

    public override void Initialize()
    {
        agentRigidbody = GetComponent<Rigidbody>();
        if (agentRigidbody == null)
        {
            agentRigidbody = gameObject.AddComponent<Rigidbody>();
        }
        agentRigidbody.freezeRotation = true;
        agentRigidbody.constraints = RigidbodyConstraints.FreezeRotationX | RigidbodyConstraints.FreezeRotationZ;

        Renderer planeRenderer = plane.GetComponent<Renderer>();
        if (planeRenderer != null)
        {
            planeRenderer.material.color = new Color(0.23f, 0.23f, 0.23f, 1f); 
        }
        
        //debugSideChannel = new DebugSideChannel();
        //SideChannelManager.RegisterSideChannel(debugSideChannel);
    }

    private HashSet<Vector3> obstacles = new HashSet<Vector3>
    {
        new Vector3(1f, 0f, 6f),
        new Vector3(1f, 0f, 7f),
        new Vector3(1f, 0f, 8f),
        new Vector3(2f, 0f, 8f),
        new Vector3(3f, 0f, 8f),
        new Vector3(4f, 0f, 8f),
        new Vector3(4f, 0f, 9f),

        new Vector3(4f, 0f, 5f),
        new Vector3(5f, 0f, 4f),

        new Vector3(3f, 0f, 1f),
        new Vector3(2f, 0f, 1f),
        new Vector3(2f, 0f, 2f),
        new Vector3(2f, 0f, 3f),

        new Vector3(7f, 0f, 6f),       
        new Vector3(7f, 0f, 7f),   
        new Vector3(7f, 0f, 8f), 
        new Vector3(6f, 0f, 8f), 
    };

    public override void OnEpisodeBegin()
    {
        Vector3 testPosition = new Vector3(9.5f, .5f, 9.5f);
        Vector3 testAngle = new Vector3(0f, 180f, 0f);
        timeStep = 0;

        transform.localPosition = testPosition;
        transform.localEulerAngles = testAngle;

        Renderer planeRenderer = plane.GetComponent<Renderer>();
        if (planeRenderer != null)
        {
            planeRenderer.material.color = new Color(0.23f, 0.23f, 0.23f, 1f); 
        }
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(transform.localPosition);
        sensor.AddObservation(timeStep);
        int x = (int)transform.localPosition[0];
        int z = (int)transform.localPosition[2];        
        float[] steptrace = System.Array.ConvertAll(otherAgent.stepTrace.GetSurroundingSteps(x, z), item => (float)item);
        sensor.AddObservation(steptrace);
        // Debug.LogFormat("x {0} z {1}", x, z);
        // DebugLogArray(otherAgent.stepTrace.GetSteps());
        // DebugStepTrace(steptrace);
    }

    void DebugStepTrace(float[] arr)
    {
        string arrayOutput = "";
        for (int i = 0; i < 8; i++) 
        {
            arrayOutput += arr[i] + "\t";
        }
        arrayOutput += "\n";
        Debug.Log(arrayOutput);
    }

    void DebugLogArray(int[,] array)
    {
        string arrayOutput = "";
        for (int i = 0; i < array.GetLength(0); i++)  
        {
            for (int j = 0; j < array.GetLength(1); j++)  
            {
                arrayOutput += array[i, j] + "\t";  
            }
            arrayOutput += "\n";  
        }
        Debug.Log(arrayOutput);  
    }

    public override void OnActionReceived(ActionBuffers actions)
    {

        if (Time.time < nextMoveTime)
        {
            return;
        }
        int action = actions.DiscreteActions[0];

        stepTrace.IncrementAll();
        Debug.LogFormat("seeker x {0}", transform.localPosition[0]);
        stepTrace.UpdateSteps((int)transform.localPosition[0], (int)transform.localPosition[2]);

        timeStep += 1;

        nextMoveTime = Time.time + moveCooldown;
        Vector3 currentPosition = transform.localPosition;
        float moveStep = 1f;
        Vector3 moveDirection = Vector3.zero;

        switch (action)
        {
            case 1:
                moveDirection = Vector3.right;
                currentPosition.x += moveStep;
                break;
            case 2:
                moveDirection = Vector3.left;
                currentPosition.x -= moveStep;
                break;
            case 3:
                moveDirection = Vector3.forward;
                currentPosition.z += moveStep;
                break;
            case 4:
                moveDirection = Vector3.back;
                currentPosition.z -= moveStep;
                break;
        }

        Vector3 zeroAdjustedCurrentPosition = currentPosition;
        zeroAdjustedCurrentPosition.x -= .5f;
        zeroAdjustedCurrentPosition.y -= .5f;
        zeroAdjustedCurrentPosition.z -= .5f;
        zeroAdjustedCurrentPosition.x = (float)Math.Round(zeroAdjustedCurrentPosition.x);
        zeroAdjustedCurrentPosition.y = (float)Math.Round(zeroAdjustedCurrentPosition.y);
        zeroAdjustedCurrentPosition.z = (float)Math.Round(zeroAdjustedCurrentPosition.z);
        bool contains = ContainsVector3(obstacles, zeroAdjustedCurrentPosition);
        if (!contains)
        {
            if (zeroAdjustedCurrentPosition.x >= 0 && zeroAdjustedCurrentPosition.x <= 9 && zeroAdjustedCurrentPosition.z >= 0 && zeroAdjustedCurrentPosition.z <= 9)
            {
                transform.localPosition = new Vector3(zeroAdjustedCurrentPosition.x + 0.5f, zeroAdjustedCurrentPosition.y + 0.5f, zeroAdjustedCurrentPosition.z + 0.5f);
                if (moveDirection != Vector3.zero)
                {
                    transform.rotation = Quaternion.LookRotation(moveDirection);
                }
            }
        }

        bool isOtherAgentVisible = IsOtherAgentVisible();
        if (isOtherAgentVisible)
        {
            Debug.Log("Found");
            StartCoroutine(ChangePlaneColorTemporarily(Color.red, .5f));
            SetReward(1.0f);
            otherAgent.Eliminate();
            EndEpisode();
        }
        if ( (timeStep + 1) % 50 == 0) {
            Debug.Log("Out of time");
            stepTrace.Reset();
            SetReward(-1.0f);
            EndEpisode();
        }
    }

    public bool IsOtherAgentVisible()
    {
        Vector3 agent_position = transform.position;
        Vector3 agent_angle = transform.localEulerAngles;
        Vector3 agent_direction = Quaternion.Euler(agent_angle) * Vector3.forward;

        Vector3 enemy_position = otherAgent.transform.position;
        Vector3 rayDirection = (enemy_position - agent_position).normalized;
        float angleToTarget = Vector3.Angle(agent_direction, rayDirection);
        float halfAngle = viewAngle / 2f;

        if (angleToTarget <= halfAngle) 
        {
            bool hits_wall = Physics.Raycast(agent_position, rayDirection, out RaycastHit wall_ray, viewDistance, obstacleMask);
            bool hits_agent = Physics.Raycast(agent_position, rayDirection, out RaycastHit agent_ray, viewDistance, agentMask);
            if (hits_wall && hits_agent)
            {
                if (Vector3.Distance(agent_position, wall_ray.point) > Vector3.Distance(agent_position, agent_ray.point)) 
                {
                    return true;
                }
            }
            else if (hits_agent)
            {
                return true;
            }
        }
        return false;
    }

    public void Eliminate()
    {
        SetReward(-1.0f);
        EndEpisode();
    }

    public void Tie()
    {
        SetReward(0.0f);
        EndEpisode();
    }

    private IEnumerator ChangePlaneColorTemporarily(Color newColor, float duration)
    {
        if (plane != null)
        {
            Renderer planeRenderer = plane.GetComponent<Renderer>();
            if (planeRenderer != null)
            {
                Color originalColor = planeRenderer.material.color; 
                planeRenderer.material.color = newColor; 
                yield return new WaitForSeconds(duration);
                planeRenderer.material.color = originalColor; 
            }
        }
    }

    private bool ContainsVector3(HashSet<Vector3> set, Vector3 value, float tolerance = 0.01f)
    {
        foreach (Vector3 vec in set)
        {
            if (Vector3.Equals(vec, value))
            {
                return true;
            }
            if (Vector3.Distance(vec, value) < tolerance)
            {
                return true;
            }
        }
        return false;
    }

    public override void Heuristic(in ActionBuffers actionBuffers)
    {
        var discreteActions = actionBuffers.DiscreteActions;
        discreteActions[0] = 0;
        if (Input.GetKey(KeyCode.D))
        {
            discreteActions[0] = 1;
        }
        else if (Input.GetKey(KeyCode.A))
        {
            discreteActions[0] = 2;
        }
        else if (Input.GetKey(KeyCode.W))
        {
            discreteActions[0] = 3;
        }
        else if (Input.GetKey(KeyCode.S))
        {
            discreteActions[0] = 4;
        }
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.TryGetComponent<Goal>(out Goal goal))
        {
            SetReward(1.0f);
            otherAgent.Eliminate();
            EndEpisode();
        }
        if (other.TryGetComponent<Hider>(out Hider hider))
        {
            Debug.Log("Contact");
            SetReward(1.0f);
            otherAgent.Eliminate();
            EndEpisode();
        }
    }
}