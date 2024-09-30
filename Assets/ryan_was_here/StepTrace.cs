using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using System.Diagnostics;  // Add this for StackTrace
using Debug = UnityEngine.Debug;  // Explicitly define Debug as UnityEngine.Debug

public class StepTrace
{
    int[,] steps;

    public StepTrace()
    {
        Reset();
    }

    public void Reset() {
        steps = new int[10, 10];
        for (int x = 0; x < 10; x++)
        {
            for (int z = 0; z < 10; z++)
            {
                steps[x, z] = int.MaxValue;
            }
        }
    }

    public void IncrementAll() {
        for (int x = 0; x < 10; x++)
        {
            for (int z = 0; z < 10; z++)
            {
                if (steps[x, z] != int.MaxValue) 
                {
                    steps[x, z]++;
                }
            }
        }
    }

    public void UpdateSteps(int x,int z)
    {
        steps[x,z] = 0;
    }

    public int[,] GetStepTrace() {
        return steps;
    }

    private bool InBounds(int x, int z) 
    {
        if (x < 0 || x >= 10 || z < 0 || z >= 10)
        {
            return false;
        }
        return true;
    }

    public int[] GetSurroundingSteps(int x, int z)
    {
        int[] vision = new int[8];
        vision[0] = InBounds(x-1, z-1) ? steps[x,z] : int.MaxValue;
        vision[1] = InBounds(x-1, z) ? steps[x,z] : int.MaxValue;
        vision[2] = InBounds(x-1, z+1) ? steps[x,z] : int.MaxValue;
        vision[3] = InBounds(x, z-1) ? steps[x,z] : int.MaxValue;
        vision[4] = InBounds(x, z+1) ? steps[x,z] : int.MaxValue;
        vision[5] = InBounds(x+1, z-1) ? steps[x,z] : int.MaxValue;
        vision[6] = InBounds(x+1, z) ? steps[x,z] : int.MaxValue;
        vision[7] = InBounds(x+1, z+1) ? steps[x,z] : int.MaxValue;
        return vision;
    }

}
