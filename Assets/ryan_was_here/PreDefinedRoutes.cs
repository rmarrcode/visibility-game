using UnityEngine;

public class PreDefinedRoutes
{
    public int cur_route;
    public int cur_idx;
        
    int[][] routes = new int[][]
    {
        //new int[] {4, 1, 1, 1, 1, 4, 4},                
        //new int[] {3, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4},   
        new int[] {3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4}
        //new int[] {1, 1, 3, 1} 
    };

    public PreDefinedRoutes()
    {
        ResetRoute();
    }

    public void ResetRoute()
    {
        cur_route = UnityEngine.Random.Range(0, routes.Length);
        cur_idx = 0;
    }

    public int NextAction()
    {
        if (cur_idx == routes[cur_route].Length)
        {
            return 0;
        }
        int next_action = routes[cur_route][cur_idx];
        cur_idx++;
        return next_action;
    }

}
