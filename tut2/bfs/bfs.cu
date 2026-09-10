#include <stdio.h>
#include <cuda_runtime.h>
#include <limits.h>
using namespace std;

__global__ void bfs(int*edges,int*idx,int*dist,bool* changed,int n,int e,int level){
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	if (u>=n || dist[u]!=level) 
		return;
	int start = idx[u];
	int end = (u+1>=n) ? e : idx[u+1];
	for (int i=start;i<end;i++){
		int v = edges[i];
		if (dist[v]!=-1)
			continue;
		dist[v] = level+1;
		*changed = true;
	}
}
int main(){
	int n, e;
	// number of vertices
	scanf("%d",&n);
	// number of edges
	scanf("%d",&e);
	int *h_dist = (int*)malloc(sizeof(int)*n);
	int *h_edges = (int*)malloc(sizeof(int)*e);
	int *h_idx = (int*)malloc(sizeof(int)*n);
	bool h_changed = false;
	// input : number of vertices
	// 		   number of neighbors of each vertex and their neighbors
	int e_idx = 0;
	for (int u=0;u<n;u++){
		h_dist[u] = -1;
		int nei;
		// number of neighbors of u
		scanf("%d",&nei);
		h_idx[u] = e_idx;
		for (int i=0;i<nei;i++){
			int v;
			// vertex
			scanf("%d",&v);
			h_edges[e_idx++] = v;
		}
	}
	h_dist[0] = 0;

	int *d_dist, *d_edges, *d_idx;
	bool *d_changed;
	cudaMalloc(&d_changed, sizeof(bool));
	cudaMalloc(&d_dist, sizeof(int)*n);
	cudaMalloc(&d_edges, sizeof(int)*e);
	cudaMalloc(&d_idx, sizeof(int)*n);
	cudaMemcpy(d_dist, h_dist, sizeof(int)*n, cudaMemcpyHostToDevice);
	cudaMemcpy(d_idx, h_idx, sizeof(int)*n, cudaMemcpyHostToDevice);
	cudaMemcpy(d_edges, h_edges, sizeof(int)*e, cudaMemcpyHostToDevice);
	cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice);

	int level = 0;
	int tpb = 1024;
	int blocks = (n+1024-1)/1024;

	// time interval calculation
	cudaEvent_t start,stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	do{
		h_changed = false;
		cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice);
		// call kernel on current level
		bfs<<<blocks,tpb>>>(d_edges,d_idx,d_dist,d_changed,n,e,level);
		cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost); 
		level++;
	}while (h_changed);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float ms;
	cudaEventElapsedTime(&ms,start,stop);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("Time elapsed: %f ms\n",ms);

	cudaMemcpy(h_dist, d_dist, sizeof(int)*n, cudaMemcpyDeviceToHost);
	for (int i=0;i<n;i++){
		printf("Distance of vertex %d from 0: %d\n",i,h_dist[i]);
	}

	// free
	cudaFree(d_changed);
	cudaFree(d_dist);
	cudaFree(d_edges);
	cudaFree(d_idx);
	free(h_edges);
	free(h_idx);
	free(h_dist);
	
	return 0;
}
