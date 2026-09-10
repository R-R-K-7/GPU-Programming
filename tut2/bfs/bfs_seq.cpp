#include <iostream>
#include <vector>
#include <cstdlib>
#include <queue>
#include <chrono>
using namespace std;

void bfs(int* edges,int* idx,int* dist,int n,int e,int src){
	bool *visited = (bool*)calloc(n,sizeof(bool));
	queue<int> q;
	q.push(src);
	dist[src] = 0;
	while (!q.empty()){
		int sz = q.size();
		for (int i=0;i<sz;i++){
			int u = q.front();q.pop();
			if (visited[u])
				continue;
			visited[u] = true;
			int start = idx[u];
			int end = (u+1>=n) ? e : idx[u+1];
			for (int i=start;i<end;i++){
				int v = edges[i];
				if (visited[v])
					continue;
				dist[v] = dist[u]+1;
				q.push(v);
			}
		}
	}
	free(visited);
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

	// time interval calculation
	auto start = chrono::high_resolution_clock::now();

	bfs(h_edges,h_idx,h_dist,n,e,0);

	auto stop = chrono::high_resolution_clock::now();
	chrono::duration<float,milli> duration = stop - start;

	printf("Time elapsed: %f ms\n",duration.count());

	for (int i=0;i<n;i++){
		printf("Distance of vertex %d from 0: %d\n",i,h_dist[i]);
	}

	// free
	free(h_edges);
	free(h_idx);
	free(h_dist);
	
	return 0;
}
