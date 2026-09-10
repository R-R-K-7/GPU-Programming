#include <stdio.h>
#include <chrono>
#include <limits.h>
#include <cstdlib>
using namespace std;

bool relax(int*edges,int*idx,int*dist,bool*active,int u,int n,int e){
	if (dist[u]==INT_MAX || !active[u])
		return false;
	bool res = false;
	int start = idx[u];
	int end = (u+1>=n) ? 2*e : idx[u+1];
	for (int i=start;i<end;i+=2){
		int v = edges[i];
		int wt = edges[i+1];
		int new_dist = dist[u] + wt;
		if (new_dist < dist[v]){
			dist[v] = new_dist;
			res = true;
			active[v] = true;
		}		
	}
	active[u] = false;
	return res;
}

int main(){
	int n, e;
	// number of vertices
	scanf("%d",&n);
	// number of edges
	scanf("%d",&e);
	int *h_dist = (int*)malloc(sizeof(int)*n);
	int *h_edges = (int*)malloc(sizeof(int)*2*e);
	int *h_idx = (int*)malloc(sizeof(int)*n);
	bool *h_active = (bool*)malloc(sizeof(bool)*n);
	bool h_changed = false;
	// input : number of vertices
	// 		   number of neighbors of each vertex and their neighbors
	int e_idx = 0;

	
	for (int u=0;u<n;u++){
		h_dist[u] = INT_MAX;
		h_active[u] = false;
		int nei;
		// number of neighbors of u
		scanf("%d",&nei);
		h_idx[u] = e_idx;
		for (int i=0;i<nei;i++){
			int v,w;
			// vertex, weight
			scanf("%d %d",&v,&w);
			h_edges[e_idx++] = v;
			h_edges[e_idx++] = w;
		}
	}

	// always consider 0 to be the source vertex
	h_dist[0] = 0;
	h_active[0] = true;
	
	auto start = chrono::high_resolution_clock::now();

	do{
		h_changed = false;
		for (int v=0;v<n;v++)
			h_changed = relax(h_edges,h_idx,h_dist,h_active,v,n,e) || h_changed;
	}while (h_changed);

	auto stop = chrono::high_resolution_clock::now();
	chrono::duration<float,milli> duration = stop - start;

	printf("Time elapsed: %f ms\n", duration.count());

	for (int i=0;i<n;i++){
		printf("distance of vertex %d from 0: %d\n", i, h_dist[i]);
	}

	// free allocated memory
	free(h_dist);
	free(h_edges);
	free(h_idx);
	free(h_active);

	return 0;
}
