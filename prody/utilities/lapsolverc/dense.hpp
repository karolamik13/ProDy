//   MIT License

//   Copyright (c) 2018-2020 Christoph Heindl
//   Copyright (c) 2019-2020 Jack Valmadre

//   Permission is hereby granted, free of charge, to any person obtaining a copy
//   of this software and associated documentation files (the "Software"), to deal
//   in the Software without restriction, including without limitation the rights
//   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//   copies of the Software, and to permit persons to whom the Software is
//   furnished to do so, subject to the following conditions:

//   The above copyright notice and this permission notice shall be included in all
//   copies or substantial portions of the Software.

//   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//   SOFTWARE.

#include <algorithm>
#include <cmath>
#include <vector>

/**
	Min cost bipartite matching via shortest augmenting paths

	This is an O(n^3) implementation of a shortest augmenting path
	algorithm for finding min cost perfect matchings in dense
	graphs.  In practice, it solves 1000x1000 problems in around 1
	second.
  
  	cost[i][j] = cost for pairing left node i with right node j
  	Lmate[i] = index of right node that left node i pairs with
  	Rmate[j] = index of left node that right node j pairs with
	The values in cost[i][j] may be positive or negative.  To perform
	maximization, simply negate the cost[][] matrix.

	Taken from https://github.com/jaehyunp/
	Adapted by https://github.com/cheind
*/
template<class T>
void solve_dense(const std::vector< std::vector<T> > &cost, std::vector<int> &Lmate, std::vector<int> &Rmate)
{
	
	//////////////////////////////////////////////////////////////////////
	// Min cost bipartite matching via shortest augmenting paths
	//
	// This is an O(n^3) implementation of a shortest augmenting path
	// algorithm for finding min cost perfect matchings in dense
	// graphs.  In practice, it solves 1000x1000 problems in around 1
	// second.
	//
	//   cost[i][j] = cost for pairing left node i with right node j
	//   Lmate[i] = index of right node that left node i pairs with
	//   Rmate[j] = index of left node that right node j pairs with
	//
	// The values in cost[i][j] may be positive or negative.  To perform
	// maximization, simply negate the cost[][] matrix.
	//////////////////////////////////////////////////////////////////////

	typedef std::vector<T> VD;
	typedef std::vector<int> VI;

	// assumes square matrices
	const int n = int(cost.size());

	// construct dual feasible solution
	VD u(n);
	VD v(n);
	for (int i = 0; i < n; i++) {
		u[i] = cost[i][0];
		for (int j = 1; j < n; j++) u[i] = std::min(u[i], cost[i][j]);
	}
	for (int j = 0; j < n; j++) {
		v[j] = cost[0][j] - u[0];
		for (int i = 1; i < n; i++) v[j] = std::min(v[j], cost[i][j] - u[i]);
	}

	// construct primal solution satisfying complementary slackness
	Lmate = VI(n, -1);
	Rmate = VI(n, -1);
	int mated = 0;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			if (Rmate[j] != -1) continue;
			if (fabs(cost[i][j] - u[i] - v[j]) < 1e-10) {
				Lmate[i] = j;
				Rmate[j] = i;
				mated++;
				break;
			}
		}
	}

	VD dist(n);
	VI dad(n);
	VI seen(n);

	// repeat until primal solution is feasible
	while (mated < n) {

		// find an unmatched left node
		int s = 0;
		while (Lmate[s] != -1) s++;

		// initialize Dijkstra
		fill(dad.begin(), dad.end(), -1);
		fill(seen.begin(), seen.end(), 0);
		for (int k = 0; k < n; k++)
			dist[k] = cost[s][k] - u[s] - v[k];

		int j = 0;
		while (true) {

			// find closest
			j = -1;
			for (int k = 0; k < n; k++) {
				if (seen[k]) continue;
				if (j == -1 || dist[k] < dist[j]) j = k;
			}
			seen[j] = 1;

			// termination condition
			if (Rmate[j] == -1) break;

			// relax neighbors
			const int i = Rmate[j];
			for (int k = 0; k < n; k++) {
				if (seen[k]) continue;
				const T new_dist = dist[j] + cost[i][k] - u[i] - v[k];
				if (dist[k] > new_dist) {
					dist[k] = new_dist;
					dad[k] = j;
				}
			}
		}

		// update dual variables
		for (int k = 0; k < n; k++) {
			if (k == j || !seen[k]) continue;
			const int i = Rmate[k];
			v[k] += dist[k] - dist[j];
			u[i] -= dist[k] - dist[j];
		}
		u[s] += dist[j];

		// augment along path
		while (dad[j] >= 0) {
			const int d = dad[j];
			Rmate[j] = Rmate[d];
			Lmate[Rmate[j]] = j;
			j = d;
		}
		Rmate[j] = s;
		Lmate[s] = j;

		mated++;
	}
}