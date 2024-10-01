using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using System.Diagnostics;  // Add this for StackTrace
using Debug = UnityEngine.Debug;  // Explicitly define Debug as UnityEngine.Debug
using System;

public class StepTrace
{
    int[,] steps;

    public int[,] GetSteps()
    {
        return steps;
    }

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
                steps[x, z] = 50;
            }
        }
    }

    public void IncrementAll() {
        for (int x = 0; x < 10; x++)
        {
            for (int z = 0; z < 10; z++)
            {
                if (steps[x, z] != 50) 
                {
                    steps[x, z]++;
                }
            }
        }
    }

    public void UpdateSteps(int x, int z)
    {
        if (x < 0 || x >= steps.GetLength(0))
        {
            throw new IndexOutOfRangeException($"Index out of bounds: x = {x}");
        }
        if (z < 0 || z >= steps.GetLength(1))
        {
            throw new IndexOutOfRangeException($"Index out of bounds: z = {z}");
        }
        steps[x, z] = 0;
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
        vision[0] = InBounds(x-1, z-1) ? steps[x-1,z-1] : 50;
        vision[1] = InBounds(x-1, z) ? steps[x-1,z] : 50;
        vision[2] = InBounds(x-1, z+1) ? steps[x-1,z+1] : 50;
        vision[3] = InBounds(x, z-1) ? steps[x,z-1] : 50;
        vision[4] = InBounds(x, z+1) ? steps[x,z+1] : 50;
        vision[5] = InBounds(x+1, z-1) ? steps[x+1,z-1] : 50;
        vision[6] = InBounds(x+1, z) ? steps[x+1,z] : 50;
        vision[7] = InBounds(x+1, z+1) ? steps[x+1,z+1] : 50;
        return vision;
    }

}
