using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using System.Diagnostics;  // Add this for StackTrace
using Debug = UnityEngine.Debug;  // Explicitly define Debug as UnityEngine.Debug
using System;

public class StepTrace
{
    float[,] steps;

    public float[,] GetSteps()
    {
        return steps;
    }

    public StepTrace()
    {
        Reset();
    }

    public void Reset() {
        steps = new float[20, 20];
        for (int x = 0; x < 10; x++)
        {
            for (int z = 0; z < 10; z++)
            {
                steps[x, z] = 0;
            }
        }
    }

    public void IncrementAll()
    {
        for (int x = 0; x < 10; x++)
        {
            for (int z = 0; z < 10; z++)
            {
                //float decayed = steps[x, z] - 0.1f;
                //steps[x, z] = decayed > 0 ? decayed : 0;
                //steps[x, z] = 1;
            }
        }
    }

    public void UpdateSteps(int x, int z)
    {
        // if (x < 0 || x >= steps.GetLength(0))
        // {
        //     throw new IndexOutOfRangeException($"Index out of bounds: x = {x}");
        // }
        // if (z < 0 || z >= steps.GetLength(1))
        // {
        //     throw new IndexOutOfRangeException($"Index out of bounds: z = {z}");
        // }
        steps[x, z] = 1;
    }

    public float[,] GetStepTrace() {
        return steps;
    }

    private bool InBounds(int x, int z) 
    {
        if (x < 0 || x >= 20 || z < 0 || z >= 20)
        {
            return false;
        }
        return true;
    }

    public float[] GetSurroundingSteps(int x, int z)
    {
        float[] vision = new float[9];
        int it = 0;
        for (int x_off = -1; x_off <= 1; x_off++) 
        {
            for (int z_off = -1; z_off <= 1; z_off++) 
            {
                if (InBounds(x+x_off, z+z_off)) 
                {
                    vision[it] = steps[x+x_off,z+z_off];
                    steps[x+x_off,z+z_off] = 0;
                }
                else
                {
                    vision[it] = 0;
                }
                it++;
            }
        }
        return vision;
    }

}
