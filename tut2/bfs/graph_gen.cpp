#include <iostream>
#include <vector>
#include <random>
#include <array>
using namespace std;

int main(int argc, char** argv){
	if (argc < 2){
		cerr << "Usage: <executable name> <number_of_vertices>\n";
		return 1;
	}
	// get number of vertices as cmd line arg
	int n = stoi(argv[1]);
	// 20 edges per vertex
	double prob = min(1.0,20.0/n);

	mt19937 rng(42);
	uniform_real_distribution<double> rand_prob(0.0,1.0);

	vector<vector<int>> adj(n);
	int total_edges = 0;

	for (int u=0;u<n;u++){
		for (int v=0;v<n;v++){
			if (u==v) continue; // no self loops
			if (rand_prob(rng) < prob){
				adj[u].push_back(v);
				total_edges++;
			}
		}
	}

	// output in format required by sssp
	cout << n << " " << total_edges << endl;
	for (int u=0;u<n;u++){
		cout << adj[u].size() << " ";
		for (auto& v : adj[u]){
			cout << v << " ";
		}
		cout << endl;
	}

	return 0;
}
