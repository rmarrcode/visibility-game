using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using System.Diagnostics;  // Add this for StackTrace
using Debug = UnityEngine.Debug;  // Explicitly define Debug as UnityEngine.Debug

public class BreadCrumbs : MonoBehaviour
{
    public Dictionary<Agent, int[,]> agentToBreadCrumbs;
    public List<Agent> agents;

    public Hider hider;
    public Seeker seeker;
    
    StepTrace stepTrace;

    void Start()
    {
        agentToBreadCrumbs = new Dictionary<Agent, int[,]>();
        agents = new List<Agent> {hider, seeker};
        stepTrace = new StepTrace();
        for (int agent_i = 0; agent_i < agents.Count; agent_i++) 
        {
            int[,] breadcrumbs = new int[10, 10];
            for (int x = 0; x < 10; x++)
            {
                for (int y = 0; y < 10; y++)
                {
                    breadcrumbs[x, y] = int.MaxValue;
                }
            }
            agentToBreadCrumbs[agents[agent_i]] = breadcrumbs;
        }
    }

    public void reset() {
        for (int agent_i = 0; agent_i < agents.Count; agent_i++) 
        {
            int[,] breadcrumbs = new int[10, 10];
            for (int x = 0; x < 10; x++)
            {
                for (int y = 0; y < 10; y++)
                {
                    breadcrumbs[x, y] = int.MaxValue;
                }
            }
            agentToBreadCrumbs[agents[agent_i]] = breadcrumbs;
        }
    }

    public void IncrementAll(Agent agent) {
        StackTrace stackTrace = new StackTrace();
        Debug.LogFormat("IncrementAll called from:\n{0}", stackTrace);
        for (int x = 0; x < 10; x++)
        {
            for (int y = 0; y < 10; y++)
            {
                if (agentToBreadCrumbs[agent][x, y] != int.MaxValue) 
                {
                    agentToBreadCrumbs[agent][x, y] = agentToBreadCrumbs[agent][x, y] + 1;
                    Debug.LogFormat("pos{0},{1}: {2}", x, y, agentToBreadCrumbs[agent][x, y]);
                }
            }
        }
    }

    public void Update()
    {
        Vector3 hiderPosition = hider.transform.position;
        agentToBreadCrumbs[hider][(int)hiderPosition[0],(int)hiderPosition[2]] = 0;
        Vector3 seekerPosition = seeker.transform.position;
        agentToBreadCrumbs[seeker][(int)seekerPosition[0],(int)seekerPosition[2]] = 0;
    }
}
