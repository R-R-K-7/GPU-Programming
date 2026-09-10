#include <stdio.h>
#include <cuda_runtime.h>
#include <limits.h>
#include <cstdlib>
using namespace std;

__global__ void sssp(int*edges,int*idx,int*dist,bool*changed,bool*active,int n,int e){
	int u = blockIdx.x * blockDim.x + threadIdx.x; 
	if (u>=n || dist[u]==INT_MAX || !active[u]) 
		return;
	int start = idx[u];
	int end = (u+1>=n) ? 2*e : idx[u+1];
	for (int i=start;i<end;i+=2){
		int v = edges[i];
		int wt = edges[i+1];
		int new_dist = dist[u] + wt;
		int old_dist = atomicMin(&dist[v], new_dist);
		if (new_dist < old_dist){
			*changed = true;
			active[v] = true;
		}
	}
	active[u] = false;
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
	int *d_dist, *d_edges, *d_idx;
	bool *d_active;
	bool *d_changed;
	cudaMalloc(&d_changed, sizeof(bool));
	cudaMalloc(&d_dist, sizeof(int)*n);
	cudaMalloc(&d_edges, sizeof(int)*2*e);
	cudaMalloc(&d_idx, sizeof(int)*n);
	cudaMalloc(&d_active, sizeof(int)*n);
	cudaMemcpy(d_dist, h_dist, sizeof(int)*n, cudaMemcpyHostToDevice);
	cudaMemcpy(d_idx, h_idx, sizeof(int)*n, cudaMemcpyHostToDevice);
	cudaMemcpy(d_edges, h_edges, sizeof(int)*2*e, cudaMemcpyHostToDevice);
	cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice);
	cudaMemcpy(d_active, h_active, sizeof(bool)*n, cudaMemcpyHostToDevice);
	int tpb = 1024;
	int blocks = (n+tpb-1) / 1024;

	// get the time of execution in gpu
	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	do{
		h_changed = false;
		// set d_changed to false
		cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice);
		// kernel call
		sssp<<<blocks, tpb>>>(d_edges,d_idx,d_dist, d_changed, d_active, n, e);
		// synchronize
		cudaDeviceSynchronize();
		// copy d_changed to h_changed
		cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);
	}while (h_changed);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float ms;
	cudaEventElapsedTime(&ms,start,stop);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("Time elapsed: %f ms\n", ms);

	cudaMemcpy(h_dist, d_dist, sizeof(int)*n, cudaMemcpyDeviceToHost);
	for (int i=0;i<n;i++){
		printf("distance of vertex %d from 0: %d\n", i, h_dist[i]);
	}

	// free allocated memory
	cudaFree(d_dist);
	cudaFree(d_edges);
	cudaFree(d_changed);
	cudaFree(d_idx);
	cudaFree(d_active);
	free(h_dist);
	free(h_edges);
	free(h_idx);
	free(h_active);

	return 0;
}
