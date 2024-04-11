//
//
// this is the DFS search based algorithm for handling edge insertion and deletion for D-core maintenance

#include "dfsSearch.h"

DfsSearch::DfsSearch(vector<pair<uint32_t, uint32_t>> &vEdges, const vector<vector<pair<::uint32_t, ::uint32_t>>> &old_d_core_decomposition) {

    edges_ = vEdges;
    sort(edges_.begin(), edges_.end());
    auto it = unique(edges_.begin(), edges_.end());
    edges_.erase(it, edges_.end());    // deduplication


    //build the adj list
    m_ = edges_.size();
    nodes_.reserve(2 * m_);
    map<uint32_t, vector<pair<uint32_t, uint32_t> > > mpRec;
    map<uint32_t, vector<pair<uint32_t, uint32_t> > >::iterator itmpRec;
    vector<pair<uint32_t, uint32_t> >::iterator itvE;
    for (uint32_t eid = 0; eid < m_; ++eid)
    {
        const uint32_t x = edges_[eid].first;
        const uint32_t y = edges_[eid].second;

        mpRec[x].emplace_back(eid, 1);
        mpRec[y].emplace_back(eid, 2);
    }
    int iRePid = 0, max_id = 0;
    for (itmpRec = mpRec.begin(); itmpRec != mpRec.end(); ++itmpRec, ++iRePid)
    {
        nodes_.push_back(itmpRec->first);
        if (itmpRec->first > max_id)
        {
            max_id = itmpRec->first;
        }
        for (itvE = itmpRec->second.begin(); itvE != itmpRec->second.end(); ++itvE)
        {
            int iEid = itvE->first;
            if (1 == itvE->second)
            {
                edges_[iEid].first = iRePid;
            }
            else if (2 == itvE->second)
            {
                edges_[iEid].second = iRePid;
            }
        }
    }
    n_ = nodes_.size();

    //build the kmax and lmax value of vertices
    ASSERT_MSG(n_ == old_d_core_decomposition.size(), "the number of vertices is not equal to the number of vertices in the old dcore decomposition result");
    k_max.resize(n_, 0);
    l_max.resize(n_);
    for(int i = 0; i < n_; i++){
        k_max[i] = old_d_core_decomposition[i][0].first;
        max_k_max = max(max_k_max, k_max[i]);
        l_max[i].resize(k_max[i] + 1, 0);
        for (auto j : old_d_core_decomposition[i]){
            l_max[i][j.first] = j.second;
        }
    }

    // initialize adjacency arrays
    adj_in.resize(n_);
    adj_out.resize(n_);
    for (uint32_t eid = 0; eid < m_; ++eid) {
        const uint32_t v1 = edges_[eid].first;
        const uint32_t v2 = edges_[eid].second;
        adj_out[v1].push_back({v2, eid});
        adj_in[v2].push_back({v1, eid});
    }
    for (uint32_t vid = 0; vid < n_; ++vid) {
        std::sort(adj_in[vid].begin(), adj_in[vid].end(),
                  [](const ArrayEntry& ae1, const ArrayEntry& ae2) {
                      return ae1.vid < ae2.vid;
                  });
        std::sort(adj_out[vid].begin(), adj_out[vid].end(),
                  [](const ArrayEntry& ae1, const ArrayEntry& ae2) {
                      return ae1.vid < ae2.vid;
                  });
    }
}


/**
    init， vEdges is the updated graph, modified_edges is the inserted/deleted edges
**/
DfsSearch::DfsSearch(vector<pair<uint32_t, uint32_t>> &vEdges, const  vector<vector<pair<::uint32_t,::uint32_t>>> & old_d_core_decomposition,
                     vector<pair<uint32_t, uint32_t>> & modified_edges)
{

    edges_ = vEdges;
    sort(edges_.begin(), edges_.end());
    auto it = unique(edges_.begin(), edges_.end());
    edges_.erase(it, edges_.end());    // deduplication


    //build the adj list
    m_ = edges_.size();
    nodes_.reserve(2 * m_);
    map<uint32_t, vector<pair<uint32_t, uint32_t> > > mpRec;
    map<uint32_t, vector<pair<uint32_t, uint32_t> > >::iterator itmpRec;
    vector<pair<uint32_t, uint32_t> >::iterator itvE;
    for (uint32_t eid = 0; eid < m_; ++eid)
    {
        const uint32_t x = edges_[eid].first;
        const uint32_t y = edges_[eid].second;

        mpRec[x].emplace_back(eid, 1);
        mpRec[y].emplace_back(eid, 2);
    }
    int iRePid = 0, max_id = 0;
    for (itmpRec = mpRec.begin(); itmpRec != mpRec.end(); ++itmpRec, ++iRePid)
    {
        nodes_.push_back(itmpRec->first);
        if (itmpRec->first > max_id)
        {
            max_id = itmpRec->first;
        }
        for (itvE = itmpRec->second.begin(); itvE != itmpRec->second.end(); ++itvE)
        {
            int iEid = itvE->first;
            if (1 == itvE->second)
            {
                edges_[iEid].first = iRePid;
            }
            else if (2 == itvE->second)
            {
                edges_[iEid].second = iRePid;
            }
        }
    }
    n_ = nodes_.size();

    //build the kmax and lmax value of vertices
    ASSERT_MSG(n_ == old_d_core_decomposition.size(), "the number of vertices is not equal to the number of vertices in the old dcore decomposition result");
    k_max.resize(n_, 0);
    l_max.resize(n_);
    for(int i = 0; i < n_; i++){
        k_max[i] = old_d_core_decomposition[i][0].first;
        l_max[i].resize(k_max[i] + 1, 0);
        for (auto j : old_d_core_decomposition[i]){
            l_max[i][j.first] = j.second;
        }
    }
    //M_ = min(k_max[modified_edges[0].first], k_max[modified_edges[0].second]);
    //N_ = max(k_max[modified_edges[0].first], k_max[modified_edges[0].second]);

    // initialize adjacency arrays
    adj_in.resize(n_);
    adj_out.resize(n_);
    for (uint32_t eid = 0; eid < m_; ++eid) {
        const uint32_t v1 = edges_[eid].first;
        const uint32_t v2 = edges_[eid].second;
        adj_out[v1].push_back({v2, eid});
        adj_in[v2].push_back({v1, eid});
    }
    for (uint32_t vid = 0; vid < n_; ++vid) {
        std::sort(adj_in[vid].begin(), adj_in[vid].end(),
                  [](const ArrayEntry& ae1, const ArrayEntry& ae2) {
                      return ae1.vid < ae2.vid;
                  });
        std::sort(adj_out[vid].begin(), adj_out[vid].end(),
                  [](const ArrayEntry& ae1, const ArrayEntry& ae2) {
                      return ae1.vid < ae2.vid;
                  });
    }

}

/**
    @brief when maintain the kmax value of vertices with edge insertion, remove disqualified vertices
    @param candidate degree of vertices, bool: whether the vertex is deleted, uint32_t: the smaller k_max value of inserted/deleted edge before graph modification, uint32_t: current vertex id
           M/N: the smaller/bigger k_max value of modified edge before graph modification
**/

void DfsSearch::kMaxRemove (vector<uint32_t> &m_d, vector<bool> &vbDeleted, uint32_t cur_node, const uint32_t &M_){
    vbDeleted[cur_node] = true;
    for(auto out_neighbor : adj_out[cur_node]){
        if(k_max[out_neighbor.vid] == M_){
            m_d[out_neighbor.vid]--;
            if(m_d[out_neighbor.vid] == M_ && !vbDeleted[out_neighbor.vid]){
                kMaxRemove(m_d, vbDeleted, out_neighbor.vid, M_);
            }
        }
    }
}

/**
    @brief when maintain the kmax value of vertices with edge deletion, using BFS to find in-core group of the endpoints of the deleted edges
    @param m_d:candidate degree of vertices, uint32_t M: the smaller k_max value of inserted/deleted edge before graph modification
           be_in_incore: whether the vertex is belongs to the in-core group,
           sub_adj_in/out: the adj list of the found in_core subgraph, uint32_t root_node_id: current vertex id
           M/N: the smaller/bigger k_max value of modified edge before graph modification
**/
void DfsSearch::kMaxFindIncore(uint32_t root_node_id, vector<uint32_t> &m_d,
                               vector<vector<DfsSearch::ArrayEntry>> &sub_adj_in,
                               vector<vector<DfsSearch::ArrayEntry>> &sub_adj_out, vector<bool> &be_in_incore,
                               const uint32_t &M_) {
    /*init*/
    list<uint32_t> lsQ;
    vector<bool> vbVisited(n_, false);
    lsQ.push_back(root_node_id);
    vbVisited[root_node_id] = true;


    while (!lsQ.empty()){
        uint32_t cur_node = lsQ.front();
        lsQ.pop_front();
        be_in_incore[cur_node] = true;
        for(auto in_neighbor : adj_in[cur_node]){
            if(k_max[in_neighbor.vid] >= M_ /*|| (k_max[in_neighbor.vid] == M_ && mED[in_neighbor.vid] > M_)*/ ){
                m_d[cur_node]++;
                if(k_max[in_neighbor.vid] == M_ && !vbVisited[in_neighbor.vid]){
                    lsQ.push_back(in_neighbor.vid);
                    vbVisited[in_neighbor.vid] = true;
                    sub_adj_in[cur_node].push_back({in_neighbor.vid, in_neighbor.eid});
                    sub_adj_out[in_neighbor.vid].push_back({cur_node, in_neighbor.eid});
                }
            }
        }
    }

}

/**
    @brief maintain the kmax value of vertices using DFS search, is_insert = true means insertion, otherwise deletion
    @param inserted/deleted edges, bool: is_insertion, M/N: the smaller/bigger k_max value of the endpoints of modified edge before graph modification
**/

void DfsSearch::maintainKmax( const vector<pair<uint32_t, uint32_t>> &modified_edges, bool is_insert, const uint32_t &M_) {
    //edge insertion
    //for the basic DFS algorithm, we only consider single edge modification
    ASSERT_MSG(modified_edges.size()==1, "the number of modified edges is not equal to 1: " << modified_edges.size());
    if (is_insert) {
        for (auto &edge: modified_edges) {
            //vector<uint32_t> dif_kmax_M_group; // the set of vertices have their kmax changed after Kmax value maintenance
            uint32_t root = edge.first;
            if (k_max[edge.second] < k_max[edge.first]) {
                root = edge.second;
            }
            list<uint32_t> lsQ;
            vector<bool> vbVisited(n_, false);
            vector<bool> vbDeleted(n_, false);
            vector<uint32_t> m_d(n_, 0);    //candidate degree of vertices
            vector<uint32_t> mED(n_, 0), mPED(n_, 0);
            /*calculate ED value of vertices*/
            for (uint32_t vid = 0; vid < n_; ++vid) {
                for (auto neighbors: adj_in[vid]) {
                    if (k_max[neighbors.vid] >= k_max[vid]) {
                        mED[vid]++;
                    }
                }
            }
            /*calculate PED value of vertices*/
            for (uint32_t vid = 0; vid < n_; ++vid) {
                for (auto neighbors: adj_in[vid]) {
                    if (k_max[neighbors.vid] > k_max[vid] ||
                        (k_max[neighbors.vid] == k_max[vid] && mED[neighbors.vid] > k_max[vid])) {
                        mPED[vid]++;
                    }
                }
            }
            m_d[root] = mPED[root], vbVisited[root] = true, lsQ.push_back(root);
            while (!lsQ.empty()) {
                uint32_t cur_node = lsQ.front();
                lsQ.pop_front();
                if (m_d[cur_node] > M_) {
                    for (auto out_neighbor: adj_out[cur_node]) {
                        if (k_max[out_neighbor.vid] == M_
                            && !vbVisited[out_neighbor.vid]
                            && mED[out_neighbor.vid] > M_) {
                            lsQ.push_back(out_neighbor.vid);
                            m_d[out_neighbor.vid] = m_d[out_neighbor.vid] + mPED[out_neighbor.vid];
                            vbVisited[out_neighbor.vid] = true;
                        }
                    }
                } else {
                    if (!vbDeleted[cur_node]) {
                        kMaxRemove(m_d, vbDeleted, cur_node, M_);
                    }

                }
            }
            for (uint32_t vid = 0; vid < n_; ++vid) {
                if (vbVisited[vid] && !vbDeleted[vid]) {
                    k_max[vid]++;
                    max_k_max = max(max_k_max, k_max[vid]);
                    l_max[vid].push_back(0);
                }
            }
        }

    }
    //edge deletion
    else{
        for (const auto &edge: modified_edges) {
            /*find the old C_{M}(G)*/
            //vector<uint32_t> dif_kmax_M_group; // the set of vertices have their kmax changed after Kmax value maintenance

            uint32_t root = edge.first;
            if (k_max[edge.second] < k_max[edge.first]) {
                root = edge.second;
            }

            /*find in-core of deleted edges*/
            vector<uint32_t> m_d(n_, 0);    //candidate degree of vertices
            vector<vector<ArrayEntry>> sub_adj_in(n_), sub_adj_out(n_);  //record structure of incore graph
            vector<bool> be_in_incore(n_, false);
            if(k_max[edge.second] == k_max[edge.first]){
                kMaxFindIncore(edge.first,  m_d, sub_adj_in, sub_adj_out, be_in_incore, M_);
                kMaxFindIncore(edge.second , m_d, sub_adj_in, sub_adj_out, be_in_incore, M_);
            }
            else{
                kMaxFindIncore(root ,m_d, sub_adj_in, sub_adj_out, be_in_incore, M_);
            }
            /*evict disqualified vertices using bin-sort, similar to decomposition algo*/
            vector<unordered_set<uint32_t>> buckets(*max_element(m_d.begin(), m_d.end()) + 1);
            vector<bool> deleted(m_, false);
            for(uint32_t vid = 0; vid < n_; ++vid){
                if(be_in_incore[vid]){
                    ASSERT_MSG(m_d[vid] >= 0, "m_d[vid] is out of range: " << m_d[vid]);
                    buckets[m_d[vid]].insert(vid);
                }
            }
            for (::uint32_t k = 0; k < buckets.size(); ++k) {
                if(buckets[k].empty()){
                    continue;
                }
                if(k >= M_){
                    break;
                }
                for (auto vid: buckets[k]) {
                    if(be_in_incore[vid]){
                        if(m_d[vid] < M_){
                            if(max_k_max == k_max[vid] && max_k_max > 0){
                                --max_k_max;
                            }
                            k_max[vid]--;
                            l_max[vid].pop_back();
                            //dif_kmax_M_group.push_back(vid);
                            for (auto out_neighbor: sub_adj_out[vid]) {
                                if(!deleted[out_neighbor.eid]){
                                    deleted[out_neighbor.eid] = true;
                                }
                                if(m_d[out_neighbor.vid] > m_d[vid]){
                                    buckets[m_d[out_neighbor.vid]].erase(out_neighbor.vid);
                                    m_d[out_neighbor.vid]--;
                                    buckets[m_d[out_neighbor.vid]].insert(out_neighbor.vid);
                                }
                            }
                            for(auto in_neighbor : sub_adj_in[vid]){
                                if(!deleted[in_neighbor.eid]){
                                    deleted[in_neighbor.eid] = true;
                                }
                            }
                        } else{
                            break;
                        }
                    }
                }
            }
        }
    }
}

/**
    @brief return h-index of a vector
    @param vector: a vector of uint32_t
**/

uint32_t DfsSearch::hIndex(const vector<uint32_t> &input_vector) {
    int n = input_vector.size();
    vector <int> bucket(n + 1);
    for(int i = 0; i < n; i++){
        int x = input_vector[i];
        if(x >= n){
            bucket[n]++;
        } else {
            bucket[x]++;
        }
    }
    int cnt = 0;
    for(int i = n; i >= 0; i--){
        cnt += bucket[i];
        if(cnt >= i)return i;
    } return -1;
}

/**
    @brief when maintain the lmax value of vertices with edge insertion, remove disqualified vertices
    @param candidate degree of vertices, bool: whether the vertex is deleted, uint32_t: the smaller k_max value of inserted/deleted edge before graph modification,
            uint32_t: current vertex id, vector<vector<ArrayEntry>>: the adj list of the (k,0)-core
**/
void DfsSearch::lMaxRemove (vector<uint32_t> &m_d, vector<uint32_t> &in_degree, vector<bool> &vbDeleted, uint32_t cur_node,
                            vector<vector<ArrayEntry>> &k_adj_in, vector<vector<ArrayEntry>> &k_adj_out, int &k, uint32_t &k_M_){
    vbDeleted[cur_node] = true;
    for(auto in_neighbor : k_adj_in[cur_node]){
        if(l_max[in_neighbor.vid][k] == k_M_ ){
            m_d[in_neighbor.vid]--;
            if(m_d[in_neighbor.vid] == k_M_ && !vbDeleted[in_neighbor.vid]){
                lMaxRemove(m_d, in_degree, vbDeleted, in_neighbor.vid, k_adj_in, k_adj_out, k, k_M_);
            }
        }
    }

    for(auto out_neighbor : k_adj_out[cur_node]){
        --in_degree[out_neighbor.vid];
        if(in_degree[out_neighbor.vid] < k && !vbDeleted[out_neighbor.vid]){
            lMaxRemove(m_d, in_degree, vbDeleted, out_neighbor.vid, k_adj_in, k_adj_out, k, k_M_);
        }
    }
}

/**
    @brief when maintain the lmax value of vertices with edge deletion, using BFS to find out-core group of the endpoints of the deleted edges
    @param m_d:candidate degree of vertices, be_in_outcore: whether the vertex is belongs to the out-core group,
           sub_adj_in/out: the adj list of the found out_core subgraph, uint32_t root_node_id: current vertex id
**/
void DfsSearch::lMaxFindOutcore(uint32_t root_node_id, vector<uint32_t> &m_d,
                               vector<vector<DfsSearch::ArrayEntry>> &sub_adj_in,
                               vector<vector<DfsSearch::ArrayEntry>> &sub_adj_out, vector<bool> &be_in_outcore,
                               uint32_t &k, uint32_t &k_M_, vector<vector<ArrayEntry>> &k_adj_out) {
    /*init*/
    list<uint32_t> lsQ;
    vector<bool> vbVisited(n_, false);
    lsQ.push_back(root_node_id);
    vbVisited[root_node_id] = true;


    while (!lsQ.empty()){
        uint32_t cur_node = lsQ.front();
        lsQ.pop_front();
        be_in_outcore[cur_node] = true;
        for(auto out_neighbor : k_adj_out[cur_node]){
            if(l_max[out_neighbor.vid][k] >= k_M_ /*|| (k_max[in_neighbor.vid] == M_ && mED[in_neighbor.vid] > M_)*/ ){
                m_d[cur_node]++;
                if(l_max[out_neighbor.vid][k] == k_M_ && !vbVisited[out_neighbor.vid]){
                    lsQ.push_back(out_neighbor.vid);
                    vbVisited[out_neighbor.vid] = true;
                    sub_adj_out[cur_node].push_back({out_neighbor.vid, out_neighbor.eid});
                    sub_adj_in[out_neighbor.vid].push_back({cur_node, out_neighbor.eid});
                }
            }
        }
    }

}

/**
    @brief maintain the klists value of vertices based on the updated kmax value, is_insert = true means insertion, otherwise deletion,
    @param inserted/deleted edges, bool: is_insertion, M/N: the smaller/bigger k_max value of the endpoints of modified edge before graph modification
**/

void DfsSearch::maintainKlist(const vector<pair<uint32_t, uint32_t>> &modified_edges, bool is_insert, const uint32_t &M_, bool k0core_pruning,
                              bool use_hindex, const int &lmax_number_of_threads,
                              const bool &reuse_pruning, const bool &skip_pruning) {
    //edge insertion
    //for the basic DFS algorithm, we only consider single edge modification
    ASSERT_MSG(modified_edges.size()==1, "the number of modified edges is not equal to 1: " << modified_edges.size());
    bool print_kocore_efficiency = true;
    if(is_insert){
        //for all the (k,0)-cores with 0 <= k <= M, we maintain the l_{max}(v, k) value of vertices
        //in the DFS search based way given edge insertion.
        if(k0core_pruning && reuse_pruning && skip_pruning){
            double out_initialzation = 0, initialzation = 0, ed_calculation = 0, dfs = 0;
            std::chrono::duration<double> final_core_update;
            auto start = omp_get_wtime();
            auto initialzation_start = omp_get_wtime();
            uint32_t dfs_search_space = 0;
            vector<vector<vector<ArrayEntry>>> total_adj_in(M_ + 1), total_adj_out(M_ + 1); //all the adj list of (k,0)-cores with 0 <= k <= M
            vector<vector<uint32_t>> total_mED_out(n_), total_mPED_out(n_), total_in_degree(n_);

            //first get the unique (k, 0)-cores
            set<uint32_t, greater<uint32_t >> k0_value_set;
            vector<uint32_t> unique_k0_cores;
            for (uint32_t i = 0; i < n_; ++i) {
                if(k_max[i] <= M_){
                    k0_value_set.insert(k_max[i]);
                }
            }
            for(uint32_t i : k0_value_set){
                unique_k0_cores.push_back(i);
            }
            if(print_kocore_efficiency) printf("k0core pruning efficiency: %f, %d, %d\n", (float)k0_value_set.size() / (float)(M_ + 1), k0_value_set.size(), M_ + 1);

            //do initialzation for reuse_pruning
            //ASSERT_MSG(lmax_number_of_threads <= M_ + 1, "reuse pruning error, the number of threads is larger than the number of (k,0)-cores " << lmax_number_of_threads << " " << M_ + 1);
            int small_batch_size = lmax_number_of_threads;
            int small_batch_number = (M_ + 1) / lmax_number_of_threads;
            int number_gap_in_batch = small_batch_number;
            if((M_ + 1) % lmax_number_of_threads != 0){  //M_ + 1 is not divisible by lmax_number_of_threads
                ++small_batch_number;
            }
            vector<vector<int>> initialization_batch( small_batch_number);
            for (int i = 0; i < lmax_number_of_threads; ++i) {
                for(int k = small_batch_number - 2; k >=0 ; --k){
                    if(k0_value_set.find(i * number_gap_in_batch + k) != k0_value_set.end()){
                        initialization_batch[k].push_back(i * number_gap_in_batch + k);
                    }
                    //initialization_batch[k].push_back(i * number_gap_in_batch + k);
                }
                if((M_ + 1) % lmax_number_of_threads == 0 ){
                    if(k0_value_set.find(i * number_gap_in_batch + small_batch_number - 1) != k0_value_set.end()){
                        initialization_batch[small_batch_number - 1].push_back(i * number_gap_in_batch + small_batch_number - 1);
                    }
                    //initialization_batch[small_batch_number - 1].push_back(i * number_gap_in_batch + small_batch_number - 1);
                }
                if( i < (M_ + 1) % lmax_number_of_threads){
                    if(k0_value_set.find((small_batch_number - 1) * lmax_number_of_threads + i) != k0_value_set.end()){
                        initialization_batch[small_batch_number - 1].push_back((small_batch_number - 1) * lmax_number_of_threads + i);
                    }
                    //initialization_batch[small_batch_number - 1].push_back((small_batch_number - 1) * lmax_number_of_threads + i);
                }

            }

            // for(int k = 0 ; k < small_batch_number; ++k){
            //     for(int i = 0; i < initialization_batch[k].size(); ++i){
            //         printf("k: %d, batch[i]: %d \n", k ,initialization_batch[k][i]);
            //     }
            // }

            for(int b = small_batch_number - 1; b >= 0; --b){
                if(b == small_batch_number - 1){
                    #pragma omp parallel for num_threads(lmax_number_of_threads)
                    for(int i = 0; i < initialization_batch[b].size(); ++i){
                        int k = initialization_batch[b][i];
                        total_adj_in[k].resize(n_);
                        total_adj_out[k].resize(n_);
                        total_mED_out[k].resize(n_, 0);
                        total_in_degree[k].resize(n_, 0);
                        for (uint32_t eid = 0; eid < edges_.size(); ++eid) {
                            const uint32_t v1 = edges_[eid].first;
                            const uint32_t v2 = edges_[eid].second;
                            if(k_max[v1] >= k && k_max[v2] >= k){
                                total_adj_in[k][v2].push_back({v1, eid});
                                total_adj_out[k][v1].push_back({v2, eid});
                                if (l_max[v2][k] >= l_max[v1][k]) {
                                    ++total_mED_out[k][v1];
                                }
                                ++total_in_degree[k][v2];
                            }
                        }
                    }

                }
                else if(b == small_batch_number - 2){
                    #pragma omp parallel for num_threads(lmax_number_of_threads)
                    for(int i = 0; i < initialization_batch[b].size(); ++i){
                        int k = initialization_batch[b][i];
                        if(find(initialization_batch.back().begin(), initialization_batch.back().end(), k+1) != initialization_batch.back().end()){
                            total_adj_in[k] = total_adj_in[k + 1];
                            total_adj_out[k] = total_adj_out[k + 1];
                            total_mED_out[k] = total_mED_out[k + 1];
                            total_in_degree[k] = total_in_degree[k + 1];
                            for (uint32_t eid = 0; eid < edges_.size(); ++eid) {
                                const uint32_t v1 = edges_[eid].first;
                                const uint32_t v2 = edges_[eid].second;
                                if(k_max[v1] >= k && k_max[v2] >= k){
                                    if(!(k_max[v1] >= k+1 && k_max[v2] >= k+1)){
                                        total_adj_in[k][v2].push_back({v1, eid});
                                        total_adj_out[k][v1].push_back({v2, eid});
                                        if (l_max[v2][k] >= l_max[v1][k]) {
                                            ++total_mED_out[k][v1];
                                        }
                                        ++total_in_degree[k][v2];
                                    }
                                }
                            }
                        }
                        else{
                            total_adj_in[k].resize(n_);
                            total_adj_out[k].resize(n_);
                            total_mED_out[k].resize(n_, 0);
                            total_in_degree[k].resize(n_, 0);
                            for (uint32_t eid = 0; eid < edges_.size(); ++eid) {
                                const uint32_t v1 = edges_[eid].first;
                                const uint32_t v2 = edges_[eid].second;
                                if(k_max[v1] >= k && k_max[v2] >= k){
                                    total_adj_in[k][v2].push_back({v1, eid});
                                    total_adj_out[k][v1].push_back({v2, eid});
                                    if (l_max[v2][k] >= l_max[v1][k]) {
                                        ++total_mED_out[k][v1];
                                    }
                                    ++total_in_degree[k][v2];
                                }
                            }
                        }
                    }
                }
                else{
                    #pragma omp parallel for num_threads(lmax_number_of_threads)
                    for(int i = 0; i < initialization_batch[b].size(); ++i){
                        int k = initialization_batch[b][i];
                        total_adj_in[k] = total_adj_in[k + 1];
                        total_adj_out[k] = total_adj_out[k + 1];
                        total_mED_out[k] = total_mED_out[k + 1];
                        total_in_degree[k] = total_in_degree[k + 1];
                        for (uint32_t eid = 0; eid < edges_.size(); ++eid) {
                            const uint32_t v1 = edges_[eid].first;
                            const uint32_t v2 = edges_[eid].second;
                            if(k_max[v1] >= k && k_max[v2] >= k){
                                if(!(k_max[v1] >= k+1 && k_max[v2] >= k+1)){
                                    total_adj_in[k][v2].push_back({v1, eid});
                                    total_adj_out[k][v1].push_back({v2, eid});
                                    if (l_max[v2][k] >= l_max[v1][k]) {
                                        ++total_mED_out[k][v1];
                                    }
                                    ++total_in_degree[k][v2];
                                }
                            }
                        }
                    }
                }
            }
            #pragma omp parallel for num_threads(lmax_number_of_threads)
            for(int k = M_ - 1; k >= 0; --k){
                if(k0_value_set.find(k) == k0_value_set.end()){
                    total_adj_in[k] = total_adj_in[k + 1];
                    total_adj_out[k] = total_adj_out[k + 1];
                    total_mED_out[k] = total_mED_out[k + 1];
                    total_in_degree[k] = total_in_degree[k + 1];
                }

            }


            vector<vector<uint32_t>> old_l_max = l_max;
            auto initialzation_end = omp_get_wtime();
            out_initialzation = initialzation_end - initialzation_start;
            #pragma omp parallel for num_threads(lmax_number_of_threads)
            for(int k : unique_k0_cores){
                /*construct adj list in the (k,0)-core*/
                //auto test1 = std::chrono::steady_clock::now();
                auto test1 = omp_get_wtime();
                vector<uint32_t> mPED_out(n_, 0);
                //auto test2 = std::chrono::steady_clock::now();
                auto test2 = omp_get_wtime();

                /*use the DFS-based method to maintain l_{max}(v,k) value of vertices*/
                uint32_t root = modified_edges[0].first;
                if (l_max[modified_edges[0].second][k] < l_max[modified_edges[0].first][k]) {
                    root = modified_edges[0].second;
                }
                uint32_t k_M_ = l_max[root][k];
                list<uint32_t> lsQ;
                vector<bool> vbVisited(n_, false);
                vector<bool> vbDeleted(n_, false);
                vector<uint32_t> m_d(n_, 0);    //candidate degree of vertices


                /*calculate PED value of vertices*/
                //#pragma omp parallel for num_threads(lmax_number_of_threads)
                for (uint32_t vid = 0; vid < n_; ++vid) {
                    if(skip_pruning && k < M_ && (k + 1 < k_max[vid]) && old_l_max[vid][k+1] == l_max[vid][k] &&
                       old_l_max[vid][k+1] != l_max[vid][k+1]){
                        vbVisited[vid] = true;
                    }
                    else if(!total_adj_out[k].empty()){
                        for (auto neighbors: total_adj_out[k][vid]) {
                            if (l_max[neighbors.vid][k] > l_max[vid][k] ||
                                (l_max[neighbors.vid][k] == l_max[vid][k] &&
                                 total_mED_out[k][neighbors.vid] > l_max[vid][k])) {
                                ++mPED_out[vid];
                            }
                        }
                    }
                }
                total_mPED_out[k] = mPED_out;

                //auto test3 = std::chrono::steady_clock::now();
                auto test3 = omp_get_wtime();
                m_d[root] = mPED_out[root], vbVisited[root] = true, lsQ.push_back(root);
                while (!lsQ.empty()) {
                    uint32_t cur_node = lsQ.front();
                    lsQ.pop_front();
                    if (m_d[cur_node] > k_M_) {
                        for (auto in_neighbor: total_adj_in[k][cur_node]) {
                            dfs_search_space++;
                            if (l_max[in_neighbor.vid][k] == k_M_
                                && !vbVisited[in_neighbor.vid]
                                && total_mED_out[k][in_neighbor.vid] > k_M_){
                                lsQ.push_back(in_neighbor.vid);
                                m_d[in_neighbor.vid] = m_d[in_neighbor.vid] + mPED_out[in_neighbor.vid];
                                vbVisited[in_neighbor.vid] = true;
                            }
                        }
                    } else {
                        if (!vbDeleted[cur_node]) {
                            lMaxRemove(m_d, total_in_degree[k], vbDeleted, cur_node, total_adj_in[k], total_adj_out[k], k, k_M_);
                        }
                    }
                }

                for (uint32_t vid = 0; vid < n_; ++vid) {
                    if (vbVisited[vid] && !vbDeleted[vid]) {
                        ++l_max[vid][k];
                    }
                }
                //auto test4 = std::chrono::steady_clock::now();
                auto test4 = omp_get_wtime();

                initialzation += test2 - test1;
                ed_calculation += test3 - test2;
                dfs += test4 - test3;
                //if(k==0) break;
            }
            printf("dfs_search_space: %u\n", dfs_search_space);

            auto end = omp_get_wtime();


            auto test1 = std::chrono::steady_clock::now();
            if(use_hindex){
                //using parallel-h-index based method to update the l_{max} value of (M_,0)-core
                for(uint32_t current_k = M_ + 1 ; current_k <= M_ + 1; ++current_k){
                    vector<vector<ArrayEntry>> k_adj_in(n_), k_adj_out(n_);
                    vector<bool> is_in_subgraph (n_, false);   //wether the vertex is in the current (M_,0)-core
                    vector<uint32_t> sub_out_coreness(n_, 0);  //the l_{max} value of vertices in the current (M_,0)-core
                    for (uint32_t eid = 0; eid < edges_.size(); ++eid) {
                        const uint32_t v1 = edges_[eid].first;
                        const uint32_t v2 = edges_[eid].second;
                        if(k_max[v1] >= current_k && k_max[v2] >= current_k){
                            k_adj_in[v2].push_back({v1, eid});
                            k_adj_out[v1].push_back({v2, eid});
                        }
                    }
                    for(uint32_t vid = 0; vid < n_; ++vid){
                        if(k_max[vid] >= current_k){
                            is_in_subgraph[vid] = true;
                            if(l_max[vid][current_k] > sub_out_coreness[vid]){
                                sub_out_coreness[vid] = l_max[vid][current_k];
                            }
                            else{
                                sub_out_coreness[vid] = k_adj_out[vid].size();
                            }
                        }
                    }
                    //parallel h-index based method
                    bool flag = true;
                    uint32_t round_cnt = 0;
                    while (flag){
                        flag = false;
#pragma omp parallel for num_threads(lmax_number_of_threads)
                        for(uint32_t vid = 0; vid < n_; ++vid){
                            if(is_in_subgraph[vid]){
                                vector<uint32_t> tmp_neighbor_out_coreness(k_adj_out[vid].size(),0);
                                for(uint32_t i = 0; i < k_adj_out[vid].size(); ++i){
                                    tmp_neighbor_out_coreness[i] = sub_out_coreness[k_adj_out[vid][i].vid];
                                }
                                uint32_t tmp_h_index = hIndex(tmp_neighbor_out_coreness);
                                if(tmp_h_index < sub_out_coreness[vid]){
                                    sub_out_coreness[vid] = tmp_h_index;
                                    flag = true;
                                }
                            }
                        }
                        round_cnt++;
                    }
                    //cout << "insert round_cnt: " << round_cnt << endl;
                    for(uint32_t vid = 0; vid < n_; ++vid){
                        if(is_in_subgraph[vid]){
                            l_max[vid][current_k] = sub_out_coreness[vid];
                        }
                    }
                }
            }
            if(!use_hindex){
                //using peeling method to update the l_{max} value of (M_+1,0)-core
                // initialize adjacency arrays for the current k-list
                for(uint32_t current_k = M_ + 1 ; current_k <= M_ + 1 ; ++current_k){
                    vector<vector<ArrayEntry>> k_adj_in, k_adj_out;
                    // 1 map the node id in k-list -> consistent, zero-based node id
                    set<::uint32_t> node_recorder;
                    map<::uint32_t,uint32_t> node_map, node_map_reverse;
                    vector<pair<uint32_t,uint32_t>> cur_k_list;
                    ::uint32_t tmp_n_ = 0;       //the number of vertices in the current k-list
                    ::uint32_t tmp_m_ = 0;          //the number of edges in the current k-list
                    for (uint32_t eid = 0; eid < edges_.size(); ++eid) {
                        const uint32_t v1 = edges_[eid].first;
                        const uint32_t v2 = edges_[eid].second;
                        if(k_max[v1] >= current_k && k_max[v2] >= current_k){
                            ++tmp_m_;
                            cur_k_list.push_back(edges_[eid]);
                            node_recorder.insert(v1);
                            node_recorder.insert(v2);
                        }
                    }
                    for (uint32_t vid : node_recorder) {
                        node_map[vid] = std::distance(node_recorder.begin(), node_recorder.find(vid));
                        node_map_reverse[node_map[vid]] = vid;    //map the consistent, zero-based node id -> node id in k-list
                    }
                    tmp_n_ = node_recorder.size();       //the number of vertices in the current k-list
                    k_adj_in.resize(tmp_n_);
                    k_adj_out.resize(tmp_n_);
                    // 2 initialize the adjacency arrays
                    for (uint32_t eid = 0; eid < cur_k_list.size(); ++eid) {
                        const uint32_t v1 = cur_k_list[eid].first;
                        const uint32_t v2 = cur_k_list[eid].second;
                        k_adj_out[node_map[v1]].push_back({node_map[v2], eid });
                        k_adj_in[node_map[v2]].push_back({node_map[v1], eid});
                    }
                    for (uint32_t vid = 0; vid < tmp_n_; ++vid) {
                        std::sort(k_adj_out[vid].begin(), k_adj_out[vid].end(),
                                  [](const ArrayEntry& ae1, const ArrayEntry& ae2) {
                                      return ae1.vid < ae2.vid;
                                  });
                        std::sort(k_adj_in[vid].begin(), k_adj_in[vid].end(),
                                  [](const ArrayEntry& ae1, const ArrayEntry& ae2) {
                                      return ae1.vid < ae2.vid;
                                  });
                    }
                    // bin-sort peeling for k-list w.r.t the out-degree, get [(k,l)-core] for all the l values
                    // note that when we delete v, for its in-neighbor w, we maintain their sub_out_coreness(a_{w}^{k})
                    // for its out-neighbor u, we maintain their in_degree in current k-list, if the in_degree < k after v's deletion
                    // then we have a_{v}^{k} = a_{u}^{k}, and we delete u as well.

                    // 1.initialization
                    std::vector<::uint32_t> deg_in(tmp_n_, 0), deg_out(tmp_n_, 0);
                    std::vector<::uint32_t> sub_out_coreness(tmp_n_, 0);
                    std::vector<bool> deleted(tmp_m_, false);  //the edge is deleted or not?
                    // 2. get the in-degree
                    ::uint32_t max_out_deg = 0;
                    for (uint32_t vid = 0; vid < tmp_n_; ++vid) {
                        deg_out[vid] = k_adj_out[vid].size();
                        max_out_deg = std::max(max_out_deg, deg_out[vid]);
                        deg_in[vid] = k_adj_in[vid].size();
                    }

                    // 3. initialize buckets
                    vector<unordered_set<::uint32_t>> buckets(max_out_deg + 1);
                    for (uint32_t vid = 0; vid < tmp_n_; ++vid) {
                        ASSERT_MSG(deg_out[vid] <= max_out_deg, "wrong sub-out-degree error " << current_k << " vid " << vid << " " << deg_out[vid] << " " << max_out_deg);
                        buckets[deg_out[vid]].insert(vid);
                    }

                    // 4. peel vertices with out-degree less than k
                    for (::uint32_t k = 0; k < buckets.size(); ++k) {
                        if(buckets[k].empty()) continue;
                        while (!buckets[k].empty()) {
                            uint32_t vid = *buckets[k].begin();
                            buckets[k].erase(buckets[k].begin());
                            sub_out_coreness[vid] = k;
                            for (const ArrayEntry &ae: k_adj_in[vid]) {   //for in neighbors, decrease their out-degree
                                ::uint32_t nbr = ae.vid;
                                if(!(deleted[ae.eid])){
                                    deleted[ae.eid] = true;
                                    if(deg_out[nbr] > k && deg_in[nbr] >= current_k) {
                                        buckets[deg_out[nbr]].erase(nbr);
                                        --deg_out[nbr];
                                        buckets[deg_out[nbr]].insert(nbr);
                                    }
                                }
                            }
                            for (const ArrayEntry &ae: k_adj_out[vid]) {  //for out neighbors, check their in-degree to see if it >= k
                                ::uint32_t nbr = ae.vid;
                                if (!(deleted[ae.eid])) {
                                    deleted[ae.eid] = true;
                                    --deg_in[nbr];
                                    if(deg_in[nbr] < current_k){
                                        //ASSERT_MSG(deg_out[nbr] >= k, "wrong out degree error " << k << " vid " << nbr << " " << deg_out[nbr] << " " << k);
                                        buckets[deg_out[nbr]].erase(nbr);
                                        buckets[k].insert(nbr);
                                    }
                                }
                            }
                        }
                    }
                    //record the (current_k, a_{u}^{current_k}) pair for all vertices
                    for (uint32_t vid = 0; vid < tmp_n_; ++vid) {
                        uint32_t u = node_map_reverse[vid];
                        l_max[u][current_k] = sub_out_coreness[vid];
                    }
                }
            }

            //update the l_{max} value of (k,0)-cores with identical structure
            if(unique_k0_cores.size() != M_ + 1){
                for(uint32_t k = unique_k0_cores.size() - 1; k > 0 ; --k){
                    if(unique_k0_cores[k] > unique_k0_cores[k-1] + 1){
                        for(uint32_t j = unique_k0_cores[k]; j > unique_k0_cores[k-1]; --j){
                            for(uint32_t vid = 0; vid < n_; ++vid){
                                if(k_max[vid] >= unique_k0_cores[k]){
                                    l_max[vid][j] = l_max[vid][unique_k0_cores[k]];
                                }
                            }
                        }
                    }
                }
            }

            auto test2 = std::chrono::steady_clock::now();
            final_core_update = test2 - test1;
            // printf("Insertion kmax initilization \x1b[1;31m%f\x1b[0m ms; ed_degree calculation costs \x1b[1;31m%f\x1b[0m ms; dfs costs \x1b[1;31m%f\x1b[0m ms;"
            //        "out initialzation costs \x1b[1;31m%f\x1b[0m ms;final_core computation costs \x1b[1;31m%f\x1b[0m ms \n",
            //         /*std::chrono::duration<double, std::milli>(initialzation).count()*/initialzation*1000,
            //         /*std::chrono::duration<double, std::milli>(ed_calculation).count()*/ed_calculation*1000,
            //         /*std::chrono::duration<double, std::milli>(dfs).count()*/dfs*1000,
            //        out_initialzation*1000,
            //        std::chrono::duration<double, std::milli>(final_core_update).count());
            // printf("total: %f ms\n", (end - start) * 1000);


        }
        else if(k0core_pruning){
            //first get the unique (k, 0)-cores
            vector<uint32_t> unique_k0_cores;
            set<uint32_t> k0_value_set;
            for (uint32_t i = 0; i < n_; ++i) {
                if(k_max[i] <= M_){
                    k0_value_set.insert(k_max[i]);
                }
            }
            for(uint32_t i : k0_value_set){
                unique_k0_cores.push_back(i);
            }
            if(print_kocore_efficiency) printf("k0core pruning efficiency: %f, %d, %d\n", (float)unique_k0_cores.size() / (float)(M_ + 1), unique_k0_cores.size(), M_ + 1);



            #pragma omp parallel for num_threads(lmax_number_of_threads)
            for(int k : unique_k0_cores){
                /*construct adj list in the (k,0)-core*/
                vector<vector<ArrayEntry>> k_adj_in, k_adj_out;
                k_adj_in.resize(n_), k_adj_out.resize(n_);
                for(uint32_t vid = 0; vid < n_; ++vid) {
                    if (k_max[vid] >= k) {
                        for (auto in_neighbor: adj_in[vid]) {
                            if (k_max[in_neighbor.vid] >= k) {
                                k_adj_in[vid].push_back({in_neighbor.vid, in_neighbor.eid});
                            }
                        }
                        for (auto out_neighbor: adj_out[vid]) {
                            if (k_max[out_neighbor.vid] >= k) {
                                k_adj_out[vid].push_back({out_neighbor.vid, out_neighbor.eid});
                            }
                        }
                    }
                }
                /*use the DFS-based method to maintain l_{max}(v,k) value of vertices*/
                uint32_t root = modified_edges[0].first;
                if (l_max[modified_edges[0].second][k] < l_max[modified_edges[0].first][k]) {
                    root = modified_edges[0].second;
                }
                uint32_t k_M_ = l_max[root][k];
                list<uint32_t> lsQ;
                vector<bool> vbVisited(n_, false);
                vector<bool> vbDeleted(n_, false);
                vector<uint32_t> m_d(n_, 0);    //candidate degree of vertices
                vector<uint32_t> mED(n_, 0), mPED(n_, 0), in_degree(n_, 0);
                /*calculate ED value of vertices*/
                for (uint32_t vid = 0; vid < n_; ++vid) {
                    if(!k_adj_out[vid].empty()){
                        for (auto neighbors: k_adj_out[vid]) {
                            if (l_max[neighbors.vid][k] >= l_max[vid][k]) {
                                mED[vid]++;
                            }
                        }
                    }
                    in_degree[vid] = k_adj_in[vid].size();
                }
                /*calculate PED value of vertices*/
                for (uint32_t vid = 0; vid < n_; ++vid) {
                    if(!k_adj_out.empty()){
                        for (auto neighbors: k_adj_out[vid]) {
                            if (l_max[neighbors.vid][k] > l_max[vid][k] ||
                                (l_max[neighbors.vid][k] == l_max[vid][k] && mED[neighbors.vid] > l_max[vid][k])) {
                                mPED[vid]++;
                            }
                        }
                    }
                }
                m_d[root] = mPED[root], vbVisited[root] = true, lsQ.push_back(root);
                while (!lsQ.empty()) {
                    uint32_t cur_node = lsQ.front();
                    lsQ.pop_front();
                    if (m_d[cur_node] > k_M_) {
                        for (auto in_neighbor: k_adj_in[cur_node]) {
                            if (l_max[in_neighbor.vid][k] == k_M_
                                && !vbVisited[in_neighbor.vid]
                                && mED[in_neighbor.vid] > k_M_) {
                                lsQ.push_back(in_neighbor.vid);
                                m_d[in_neighbor.vid] = m_d[in_neighbor.vid] + mPED[in_neighbor.vid];
                                vbVisited[in_neighbor.vid] = true;
                            }
                        }
                    } else {
                        if (!vbDeleted[cur_node]) {
                            lMaxRemove(m_d, in_degree, vbDeleted, cur_node, k_adj_in, k_adj_out, k, k_M_);
                        }
                    }
                }
                for (uint32_t vid = 0; vid < n_; ++vid) {
                    if (vbVisited[vid] && !vbDeleted[vid]) {
                        l_max[vid][k]++;
                    }
                }
            }

            if(use_hindex){
                //using parallel-h-index based method to update the l_{max} value of (M_,0)-core
                for(uint32_t current_k = M_ + 1 ; current_k <= M_ + 1; ++current_k){
                    vector<vector<ArrayEntry>> k_adj_in(n_), k_adj_out(n_);
                    vector<pair<uint32_t,uint32_t>> cur_k_list;
                    vector<bool> is_in_subgraph (n_, false);   //wether the vertex is in the current (M_,0)-core
                    vector<uint32_t> sub_out_coreness(n_, 0);  //the l_{max} value of vertices in the current (M_,0)-core
                    for (uint32_t eid = 0; eid < edges_.size(); ++eid) {
                        const uint32_t v1 = edges_[eid].first;
                        const uint32_t v2 = edges_[eid].second;
                        if(k_max[v1] >= current_k && k_max[v2] >= current_k){
                            k_adj_in[v2].push_back({v1, eid});
                            k_adj_out[v1].push_back({v2, eid});
                            cur_k_list.push_back(edges_[eid]);
                        }
                    }
                    for(uint32_t vid = 0; vid < n_; ++vid){
                        if(k_max[vid] >= current_k){
                            is_in_subgraph[vid] = true;
                            if(l_max[vid][current_k] > sub_out_coreness[vid]){
                                sub_out_coreness[vid] = l_max[vid][current_k];
                            }
                            else{
                                sub_out_coreness[vid] = k_adj_out[vid].size();
                            }
                        }
                    }
                    //parallel h-index based method
                    bool flag = true;
                    while (flag){
                        flag = false;
                        #pragma omp parallel for num_threads(lmax_number_of_threads)
                        for(uint32_t vid = 0; vid < n_; ++vid){
                            if(is_in_subgraph[vid]){
                                vector<uint32_t> tmp_neighbor_out_coreness(k_adj_out[vid].size(),0);
                                for(uint32_t i = 0; i < k_adj_out[vid].size(); ++i){
                                    tmp_neighbor_out_coreness[i] = sub_out_coreness[k_adj_out[vid][i].vid];
                                }
                                if(hIndex(tmp_neighbor_out_coreness) < sub_out_coreness[vid]){
                                    sub_out_coreness[vid] = hIndex(tmp_neighbor_out_coreness);
                                    flag = true;
                                }
                            }
                        }
                    }
                    for(uint32_t vid = 0; vid < n_; ++vid){
                        if(is_in_subgraph[vid]){
                            l_max[vid][current_k] = sub_out_coreness[vid];
                        }
                    }
                }
            }
            if(!use_hindex){
                //using peeling method to update the l_{max} value of (M_+1,0)-core with identical structure
                // initialize adjacency arrays for the current k-list
                for(uint32_t current_k = M_ + 1 ; current_k <= M_ + 1 ; ++current_k){
                    vector<vector<ArrayEntry>> k_adj_in, k_adj_out;
                    // 1 map the node id in k-list -> consistent, zero-based node id
                    set<::uint32_t> node_recorder;
                    map<::uint32_t,uint32_t> node_map, node_map_reverse;
                    vector<pair<uint32_t,uint32_t>> cur_k_list;
                    ::uint32_t tmp_n_ = 0;       //the number of vertices in the current k-list
                    ::uint32_t tmp_m_ = 0;          //the number of edges in the current k-list
                    for (uint32_t eid = 0; eid < edges_.size(); ++eid) {
                        const uint32_t v1 = edges_[eid].first;
                        const uint32_t v2 = edges_[eid].second;
                        if(k_max[v1] >= current_k && k_max[v2] >= current_k){
                            ++tmp_m_;
                            cur_k_list.push_back(edges_[eid]);
                            node_recorder.insert(v1);
                            node_recorder.insert(v2);
                        }
                    }
                    for (uint32_t vid : node_recorder) {
                        node_map[vid] = std::distance(node_recorder.begin(), node_recorder.find(vid));
                        node_map_reverse[node_map[vid]] = vid;    //map the consistent, zero-based node id -> node id in k-list
                    }
                    tmp_n_ = node_recorder.size();       //the number of vertices in the current k-list
                    k_adj_in.resize(tmp_n_);
                    k_adj_out.resize(tmp_n_);
                    // 2 initialize the adjacency arrays
                    for (uint32_t eid = 0; eid < cur_k_list.size(); ++eid) {
                        const uint32_t v1 = cur_k_list[eid].first;
                        const uint32_t v2 = cur_k_list[eid].second;
                        k_adj_out[node_map[v1]].push_back({node_map[v2], eid });
                        k_adj_in[node_map[v2]].push_back({node_map[v1], eid });
                    }
                    for (uint32_t vid = 0; vid < tmp_n_; ++vid) {
                        std::sort(k_adj_out[vid].begin(), k_adj_out[vid].end(),
                                  [](const ArrayEntry& ae1, const ArrayEntry& ae2) {
                                      return ae1.vid < ae2.vid;
                                  });
                        std::sort(k_adj_in[vid].begin(), k_adj_in[vid].end(),
                                  [](const ArrayEntry& ae1, const ArrayEntry& ae2) {
                                      return ae1.vid < ae2.vid;
                                  });
                    }
                    // bin-sort peeling for k-list w.r.t the out-degree, get [(k,l)-core] for all the l values
                    // note that when we delete v, for its in-neighbor w, we maintain their sub_out_coreness(a_{w}^{k})
                    // for its out-neighbor u, we maintain their in_degree in current k-list, if the in_degree < k after v's deletion
                    // then we have a_{v}^{k} = a_{u}^{k}, and we delete u as well.

                    // 1.initialization
                    std::vector<::uint32_t> deg_in(tmp_n_, 0), deg_out(tmp_n_, 0);
                    std::vector<::uint32_t> sub_out_coreness(tmp_n_, 0);
                    std::vector<bool> deleted(tmp_m_, false);  //the edge is deleted or not?
                    // 2. get the in-degree
                    ::uint32_t max_out_deg = 0;
                    for (uint32_t vid = 0; vid < tmp_n_; ++vid) {
                        deg_out[vid] = k_adj_out[vid].size();
                        max_out_deg = std::max(max_out_deg, deg_out[vid]);
                        deg_in[vid] = k_adj_in[vid].size();
                    }

                    // 3. initialize buckets
                    vector<unordered_set<::uint32_t>> buckets(max_out_deg + 1);
                    for (uint32_t vid = 0; vid < tmp_n_; ++vid) {
                        ASSERT_MSG(deg_out[vid] <= max_out_deg, "wrong sub-out-degree error " << current_k << " vid " << vid << " " << deg_out[vid] << " " << max_out_deg);
                        buckets[deg_out[vid]].insert(vid);
                    }

                    // 4. peel vertices with out-degree less than k
                    for (::uint32_t k = 0; k < buckets.size(); ++k) {
                        if(buckets[k].empty()) continue;
                        while (!buckets[k].empty()) {
                            uint32_t vid = *buckets[k].begin();
                            buckets[k].erase(buckets[k].begin());
                            sub_out_coreness[vid] = k;
                            for (const ArrayEntry &ae: k_adj_in[vid]) {   //for in neighbors, decrease their out-degree
                                ::uint32_t nbr = ae.vid;
                                if(!(deleted[ae.eid])){
                                    deleted[ae.eid] = true;
                                    if(deg_out[nbr] > k && deg_in[nbr] >= current_k) {
                                        buckets[deg_out[nbr]].erase(nbr);
                                        --deg_out[nbr];
                                        buckets[deg_out[nbr]].insert(nbr);
                                    }
                                }
                            }
                            for (const ArrayEntry &ae: k_adj_out[vid]) {  //for out neighbors, check their in-degree to see if it >= k
                                ::uint32_t nbr = ae.vid;
                                if (!(deleted[ae.eid])) {
                                    deleted[ae.eid] = true;
                                    --deg_in[nbr];
                                    if(deg_in[nbr] < current_k){
                                        ASSERT_MSG(deg_out[nbr] >= k, "wrong out degree error " << k << " vid " << nbr << " " << deg_out[nbr] << " " << k);
                                        buckets[deg_out[nbr]].erase(nbr);
                                        buckets[k].insert(nbr);
                                    }
                                }
                            }
                        }
                    }
                    //record the (current_k, a_{u}^{current_k}) pair for all vertices
                    for (uint32_t vid = 0; vid < tmp_n_; ++vid) {
                        uint32_t u = node_map_reverse[vid];
                        l_max[u][current_k] = sub_out_coreness[vid];
                    }
                }
            }


            //update the l_{max} value of (k,0)-cores with identical structure
            if(unique_k0_cores.size() != M_ + 1){
                for(uint32_t k = unique_k0_cores.size() - 1; k > 0 ; --k){
                    if(unique_k0_cores[k] > unique_k0_cores[k-1] + 1){
                        for(uint32_t j = unique_k0_cores[k]; j > unique_k0_cores[k-1]; --j){
                            for(uint32_t vid = 0; vid < n_; ++vid){
                                if(k_max[vid] >= unique_k0_cores[k]){
                                    l_max[vid][j] = l_max[vid][unique_k0_cores[k]];
                                }
                            }
                        }
                    }
                }
            }

        }
        else if(reuse_pruning){
            double out_initialzation = 0, initialzation = 0, ed_calculation = 0, dfs = 0;
            std::chrono::duration<double> final_core_update;
            auto start = omp_get_wtime();
            auto initialzation_start = omp_get_wtime();
            uint32_t dfs_search_space = 0;
            vector<vector<vector<ArrayEntry>>> total_adj_in(M_ + 1), total_adj_out(M_ + 1); //all the adj list of (k,0)-cores with 0 <= k <= M
            vector<vector<uint32_t>> total_mED_out(n_), total_mPED_out(n_), total_in_degree(n_);

            //ASSERT_MSG(lmax_number_of_threads <= M_ + 1, "reuse pruning error, the number of threads is larger than the number of (k,0)-cores " << lmax_number_of_threads << " " << M_ + 1);
            int small_batch_size = lmax_number_of_threads;
            int small_batch_number = (M_ + 1) / lmax_number_of_threads;
            int number_gap_in_batch = small_batch_number;
            if((M_ + 1) % lmax_number_of_threads != 0){  //M_ + 1 is not divisible by lmax_number_of_threads
                ++small_batch_number;
            }
            vector<vector<int>> initialization_batch( small_batch_number);
            for (int i = 0; i < lmax_number_of_threads; ++i) {
                for(int k = small_batch_number - 2; k >=0 ; --k){
                    initialization_batch[k].push_back(i * number_gap_in_batch + k);
                }
                if((M_ + 1) % lmax_number_of_threads == 0 ){
                    initialization_batch[small_batch_number - 1].push_back(i * number_gap_in_batch + small_batch_number - 1);
                }
                if( i < (M_ + 1) % lmax_number_of_threads){
                    initialization_batch[small_batch_number - 1].push_back((small_batch_number - 1) * lmax_number_of_threads + i);
                }

            }

            for(int b = small_batch_number - 1; b >= 0; --b){
                if(b == small_batch_number - 1){
                    #pragma omp parallel for num_threads(lmax_number_of_threads)
                    for(int i = 0; i < initialization_batch[b].size(); ++i){
                        int k = initialization_batch[b][i];
                        total_adj_in[k].resize(n_);
                        total_adj_out[k].resize(n_);
                        total_mED_out[k].resize(n_, 0);
                        total_in_degree[k].resize(n_, 0);
                        for (uint32_t eid = 0; eid < edges_.size(); ++eid) {
                            const uint32_t v1 = edges_[eid].first;
                            const uint32_t v2 = edges_[eid].second;
                            if(k_max[v1] >= k && k_max[v2] >= k){
                                total_adj_in[k][v2].push_back({v1, eid});
                                total_adj_out[k][v1].push_back({v2, eid});
                                if (l_max[v2][k] >= l_max[v1][k]) {
                                    ++total_mED_out[k][v1];
                                }
                                ++total_in_degree[k][v2];
                            }
                        }
                    }

                }
                else if(b == small_batch_number - 2){
                    #pragma omp parallel for num_threads(lmax_number_of_threads)
                    for(int i = 0; i < initialization_batch[b].size(); ++i){
                        int k = initialization_batch[b][i];
                        if(find(initialization_batch.back().begin(), initialization_batch.back().end(), k+1) != initialization_batch.back().end()){
                            total_adj_in[k] = total_adj_in[k + 1];
                            total_adj_out[k] = total_adj_out[k + 1];
                            total_mED_out[k] = total_mED_out[k + 1];
                            total_in_degree[k] = total_in_degree[k + 1];
                            for (uint32_t eid = 0; eid < edges_.size(); ++eid) {
                                const uint32_t v1 = edges_[eid].first;
                                const uint32_t v2 = edges_[eid].second;
                                if(k_max[v1] >= k && k_max[v2] >= k){
                                    if(!(k_max[v1] >= k+1 && k_max[v2] >= k+1)){
                                        total_adj_in[k][v2].push_back({v1, eid});
                                        total_adj_out[k][v1].push_back({v2, eid});
                                        if (l_max[v2][k] >= l_max[v1][k]) {
                                            ++total_mED_out[k][v1];
                                        }
                                        ++total_in_degree[k][v2];
                                    }
                                }
                            }
                        }
                        else{
                            total_adj_in[k].resize(n_);
                            total_adj_out[k].resize(n_);
                            total_mED_out[k].resize(n_, 0);
                            total_in_degree[k].resize(n_, 0);
                            for (uint32_t eid = 0; eid < edges_.size(); ++eid) {
                                const uint32_t v1 = edges_[eid].first;
                                const uint32_t v2 = edges_[eid].second;
                                if(k_max[v1] >= k && k_max[v2] >= k){
                                    total_adj_in[k][v2].push_back({v1, eid});
                                    total_adj_out[k][v1].push_back({v2, eid});
                                    if (l_max[v2][k] >= l_max[v1][k]) {
                                        ++total_mED_out[k][v1];
                                    }
                                    ++total_in_degree[k][v2];
                                }
                            }
                        }
                    }
                }
                else{
                    #pragma omp parallel for num_threads(lmax_number_of_threads)
                    for(int i = 0; i < initialization_batch[b].size(); ++i){
                        int k = initialization_batch[b][i];
                        total_adj_in[k] = total_adj_in[k + 1];
                        total_adj_out[k] = total_adj_out[k + 1];
                        total_mED_out[k] = total_mED_out[k + 1];
                        total_in_degree[k] = total_in_degree[k + 1];
                        for (uint32_t eid = 0; eid < edges_.size(); ++eid) {
                            const uint32_t v1 = edges_[eid].first;
                            const uint32_t v2 = edges_[eid].second;
                            if(k_max[v1] >= k && k_max[v2] >= k){
                                if(!(k_max[v1] >= k+1 && k_max[v2] >= k+1)){
                                    total_adj_in[k][v2].push_back({v1, eid});
                                    total_adj_out[k][v1].push_back({v2, eid});
                                    if (l_max[v2][k] >= l_max[v1][k]) {
                                        ++total_mED_out[k][v1];
                                    }
                                    ++total_in_degree[k][v2];
                                }
                            }
                        }
                    }
                }
            }

            vector<vector<uint32_t>> old_l_max = l_max;
            auto initialzation_end = omp_get_wtime();
            out_initialzation = initialzation_end - initialzation_start;

            #pragma omp parallel for num_threads(lmax_number_of_threads)
            for(int k = M_; k >= 0; --k){
                /*construct adj list in the (k,0)-core*/
                //auto test1 = std::chrono::steady_clock::now();
                auto test1 = omp_get_wtime();
                vector<uint32_t> mPED_out(n_, 0);
                //auto test2 = std::chrono::steady_clock::now();
                auto test2 = omp_get_wtime();

                /*use the DFS-based method to maintain l_{max}(v,k) value of vertices*/
                uint32_t root = modified_edges[0].first;
                if (l_max[modified_edges[0].second][k] < l_max[modified_edges[0].first][k]) {
                    root = modified_edges[0].second;
                }
                uint32_t k_M_ = l_max[root][k];
                list<uint32_t> lsQ;
                vector<bool> vbVisited(n_, false);
                vector<bool> vbDeleted(n_, false);
                vector<uint32_t> m_d(n_, 0);    //candidate degree of vertices


                /*calculate PED value of vertices*/
                for (uint32_t vid = 0; vid < n_; ++vid) {
                     if(!total_adj_out[k].empty()){
                        for (auto neighbors: total_adj_out[k][vid]) {
                            if (l_max[neighbors.vid][k] > l_max[vid][k] ||
                                (l_max[neighbors.vid][k] == l_max[vid][k] &&
                                 total_mED_out[k][neighbors.vid] > l_max[vid][k])) {
                                ++mPED_out[vid];
                            }
                        }
                    }
                }
                total_mPED_out[k] = mPED_out;

                //auto test3 = std::chrono::steady_clock::now();
                auto test3 = omp_get_wtime();
                m_d[root] = mPED_out[root], vbVisited[root] = true, lsQ.push_back(root);
                while (!lsQ.empty()) {
                    uint32_t cur_node = lsQ.front();
                    lsQ.pop_front();
                    if (m_d[cur_node] > k_M_) {
                        for (auto in_neighbor: total_adj_in[k][cur_node]) {
                            ++dfs_search_space;
                            if (l_max[in_neighbor.vid][k] == k_M_
                                && !vbVisited[in_neighbor.vid]
                                && total_mED_out[k][in_neighbor.vid] > k_M_){
                                lsQ.push_back(in_neighbor.vid);
                                m_d[in_neighbor.vid] = m_d[in_neighbor.vid] + mPED_out[in_neighbor.vid];
                                vbVisited[in_neighbor.vid] = true;
                            }
                        }
                    } else {
                        if (!vbDeleted[cur_node]) {
                            lMaxRemove(m_d, total_in_degree[k], vbDeleted, cur_node, total_adj_in[k], total_adj_out[k], k, k_M_);
                        }
                    }
                }

                for (uint32_t vid = 0; vid < n_; ++vid) {
                    if (vbVisited[vid] && !vbDeleted[vid]) {
                        ++l_max[vid][k];
                    }
                }
                //auto test4 = std::chrono::steady_clock::now();
                auto test4 = omp_get_wtime();

                initialzation += test2 - test1;
                ed_calculation += test3 - test2;
                dfs += test4 - test3;
                //if(k==0) break;
            }
            printf("dfs_search_space: %u\n", dfs_search_space);
        }
        else if(skip_pruning){
            /*std::chrono::duration<double>*/double out_initialzation = 0, initialzation = 0, ed_calculation = 0, dfs = 0;
            std::chrono::duration<double> final_core_update;
            uint32_t dfs_search_space = 0;
            vector<vector<uint32_t>> old_l_max = l_max;
            auto start = omp_get_wtime();
            /*original version, direct parallel*/
            #pragma omp parallel for num_threads(lmax_number_of_threads)
            for(int k = M_; k >= 0; --k){
                /*construct adj list in the (k,0)-core*/
                //auto test1 = std::chrono::steady_clock::now();
                auto test1 = omp_get_wtime();
                vector<vector<ArrayEntry>> k_adj_in(n_), k_adj_out(n_);
                vector<uint32_t> mED_out(n_, 0), mPED_out(n_, 0), in_degree(n_, 0);
                for (uint32_t eid = 0; eid < edges_.size(); ++eid) {
                    const uint32_t v1 = edges_[eid].first;
                    const uint32_t v2 = edges_[eid].second;
                    if(k_max[v1] >= k && k_max[v2] >= k){
                        k_adj_in[v2].push_back({v1, eid});
                        k_adj_out[v1].push_back({v2, eid});
                        if (l_max[v2][k] >= l_max[v1][k]) {
                            mED_out[v1]++;
                        }
                        ++in_degree[v2];
                    }
                }

                //auto test2 = std::chrono::steady_clock::now();
                auto test2 = omp_get_wtime();

                /*use the DFS-based method to maintain l_{max}(v,k) value of vertices*/
                uint32_t root = modified_edges[0].first;
                if (l_max[modified_edges[0].second][k] < l_max[modified_edges[0].first][k]) {
                    root = modified_edges[0].second;
                }
                uint32_t k_M_ = l_max[root][k];
                list<uint32_t> lsQ;
                vector<bool> vbVisited(n_, false);
                vector<bool> vbDeleted(n_, false);
                vector<uint32_t> m_d(n_, 0);    //candidate degree of vertices


                /*calculate PED value of vertices*/
                for (uint32_t vid = 0; vid < n_; ++vid) {
                    if(skip_pruning && k < M_ && (k + 1 < k_max[vid]) && old_l_max[vid][k+1] == l_max[vid][k] &&
                       old_l_max[vid][k+1] != l_max[vid][k+1]){
                        vbVisited[vid] = true;
                    }
                   else if(!k_adj_out.empty()){
                        for (auto neighbors: k_adj_out[vid]) {
                            if (l_max[neighbors.vid][k] > l_max[vid][k] ||
                                (l_max[neighbors.vid][k] == l_max[vid][k] && mED_out[neighbors.vid] > l_max[vid][k])) {
                                mPED_out[vid]++;
                            }
                        }
                    }
                }

                //auto test3 = std::chrono::steady_clock::now();
                auto test3 = omp_get_wtime();
                m_d[root] = mPED_out[root], vbVisited[root] = true, lsQ.push_back(root);
                while (!lsQ.empty()) {
                    uint32_t cur_node = lsQ.front();
                    lsQ.pop_front();
                    if (m_d[cur_node] > k_M_) {
                        for (auto in_neighbor: k_adj_in[cur_node]) {
                            dfs_search_space++;
                            if (l_max[in_neighbor.vid][k] == k_M_
                                && !vbVisited[in_neighbor.vid]
                                && mED_out[in_neighbor.vid] > k_M_){
                                lsQ.push_back(in_neighbor.vid);
                                m_d[in_neighbor.vid] = m_d[in_neighbor.vid] + mPED_out[in_neighbor.vid];
                                vbVisited[in_neighbor.vid] = true;
                            }
                        }
                    } else {
                        if (!vbDeleted[cur_node]) {
                            lMaxRemove(m_d, in_degree, vbDeleted, cur_node, k_adj_in, k_adj_out, k, k_M_);
                        }
                    }
                }
                for (uint32_t vid = 0; vid < n_; ++vid) {
                    if (vbVisited[vid] && !vbDeleted[vid]) {
                        ++l_max[vid][k];
                    }
                }
                //auto test4 = std::chrono::steady_clock::now();
                auto test4 = omp_get_wtime();

                initialzation += test2 - test1;
                ed_calculation += test3 - test2;
                dfs += test4 - test3;
            }
            auto end = omp_get_wtime();
            printf("dfs_search_space: %u\n", dfs_search_space);
            auto test1 = std::chrono::steady_clock::now();
            if(use_hindex){
                //using parallel-h-index based method to update the l_{max} value of (M_,0)-core
                for(uint32_t current_k = M_ + 1 ; current_k <= M_ + 1; ++current_k){
                    vector<vector<ArrayEntry>> k_adj_in(n_), k_adj_out(n_);
                    vector<pair<uint32_t,uint32_t>> cur_k_list;
                    vector<bool> is_in_subgraph (n_, false);   //wether the vertex is in the current (M_,0)-core
                    vector<uint32_t> sub_out_coreness(n_, 0);  //the l_{max} value of vertices in the current (M_,0)-core
                    for (uint32_t eid = 0; eid < edges_.size(); ++eid) {
                        const uint32_t v1 = edges_[eid].first;
                        const uint32_t v2 = edges_[eid].second;
                        if(k_max[v1] >= current_k && k_max[v2] >= current_k){
                            k_adj_in[v2].push_back({v1, eid});
                            k_adj_out[v1].push_back({v2, eid});
                            cur_k_list.push_back(edges_[eid]);
                        }
                    }
                    for(uint32_t vid = 0; vid < n_; ++vid){
                        if(k_max[vid] >= current_k){
                            is_in_subgraph[vid] = true;
                            if(l_max[vid][current_k] > sub_out_coreness[vid]){
                                sub_out_coreness[vid] = l_max[vid][current_k];
                            }
                            else{
                                sub_out_coreness[vid] = k_adj_out[vid].size();
                            }
                        }
                    }
                    //parallel h-index based method
                    bool flag = true;
                    uint32_t round_cnt = 0;
                    while (flag){
                        flag = false;
#pragma omp parallel for num_threads(lmax_number_of_threads)
                        for(uint32_t vid = 0; vid < n_; ++vid){
                            if(is_in_subgraph[vid]){
                                vector<uint32_t> tmp_neighbor_out_coreness(k_adj_out[vid].size(),0);
                                for(uint32_t i = 0; i < k_adj_out[vid].size(); ++i){
                                    tmp_neighbor_out_coreness[i] = sub_out_coreness[k_adj_out[vid][i].vid];
                                }
                                uint32_t tmp_h_index = hIndex(tmp_neighbor_out_coreness);
                                if(tmp_h_index < sub_out_coreness[vid]){
                                    sub_out_coreness[vid] = tmp_h_index;
                                    flag = true;
                                }
                            }
                        }
                        round_cnt++;
                    }
                    //cout << "insert round_cnt: " << round_cnt << endl;
                    for(uint32_t vid = 0; vid < n_; ++vid){
                        if(is_in_subgraph[vid]){
                            l_max[vid][current_k] = sub_out_coreness[vid];
                        }
                    }
                }
            }
            if(!use_hindex){
                //using peeling method to update the l_{max} value of (M_+1,0)-core
                // initialize adjacency arrays for the current k-list
                for(uint32_t current_k = M_ + 1 ; current_k <= M_ + 1 ; ++current_k){
                    vector<vector<ArrayEntry>> k_adj_in, k_adj_out;
                    // 1 map the node id in k-list -> consistent, zero-based node id
                    set<::uint32_t> node_recorder;
                    map<::uint32_t,uint32_t> node_map, node_map_reverse;
                    vector<pair<uint32_t,uint32_t>> cur_k_list;
                    ::uint32_t tmp_n_ = 0;       //the number of vertices in the current k-list
                    ::uint32_t tmp_m_ = 0;          //the number of edges in the current k-list
                    for (uint32_t eid = 0; eid < edges_.size(); ++eid) {
                        const uint32_t v1 = edges_[eid].first;
                        const uint32_t v2 = edges_[eid].second;
                        if(k_max[v1] >= current_k && k_max[v2] >= current_k){
                            ++tmp_m_;
                            cur_k_list.push_back(edges_[eid]);
                            node_recorder.insert(v1);
                            node_recorder.insert(v2);
                        }
                    }
                    for (uint32_t vid : node_recorder) {
                        node_map[vid] = std::distance(node_recorder.begin(), node_recorder.find(vid));
                        node_map_reverse[node_map[vid]] = vid;    //map the consistent, zero-based node id -> node id in k-list
                    }
                    tmp_n_ = node_recorder.size();       //the number of vertices in the current k-list
                    k_adj_in.resize(tmp_n_);
                    k_adj_out.resize(tmp_n_);
                    // 2 initialize the adjacency arrays
                    for (uint32_t eid = 0; eid < cur_k_list.size(); ++eid) {
                        const uint32_t v1 = cur_k_list[eid].first;
                        const uint32_t v2 = cur_k_list[eid].second;
                        k_adj_out[node_map[v1]].push_back({node_map[v2], eid });
                        k_adj_in[node_map[v2]].push_back({node_map[v1], eid});
                    }
                    for (uint32_t vid = 0; vid < tmp_n_; ++vid) {
                        std::sort(k_adj_out[vid].begin(), k_adj_out[vid].end(),
                                  [](const ArrayEntry& ae1, const ArrayEntry& ae2) {
                                      return ae1.vid < ae2.vid;
                                  });
                        std::sort(k_adj_in[vid].begin(), k_adj_in[vid].end(),
                                  [](const ArrayEntry& ae1, const ArrayEntry& ae2) {
                                      return ae1.vid < ae2.vid;
                                  });
                    }
                    // bin-sort peeling for k-list w.r.t the out-degree, get [(k,l)-core] for all the l values
                    // note that when we delete v, for its in-neighbor w, we maintain their sub_out_coreness(a_{w}^{k})
                    // for its out-neighbor u, we maintain their in_degree in current k-list, if the in_degree < k after v's deletion
                    // then we have a_{v}^{k} = a_{u}^{k}, and we delete u as well.

                    // 1.initialization
                    std::vector<::uint32_t> deg_in(tmp_n_, 0), deg_out(tmp_n_, 0);
                    std::vector<::uint32_t> sub_out_coreness(tmp_n_, 0);
                    std::vector<bool> deleted(tmp_m_, false);  //the edge is deleted or not?
                    // 2. get the in-degree
                    ::uint32_t max_out_deg = 0;
                    for (uint32_t vid = 0; vid < tmp_n_; ++vid) {
                        deg_out[vid] = k_adj_out[vid].size();
                        max_out_deg = std::max(max_out_deg, deg_out[vid]);
                        deg_in[vid] = k_adj_in[vid].size();
                    }

                    // 3. initialize buckets
                    vector<unordered_set<::uint32_t>> buckets(max_out_deg + 1);
                    for (uint32_t vid = 0; vid < tmp_n_; ++vid) {
                        ASSERT_MSG(deg_out[vid] <= max_out_deg, "wrong sub-out-degree error " << current_k << " vid " << vid << " " << deg_out[vid] << " " << max_out_deg);
                        buckets[deg_out[vid]].insert(vid);
                    }

                    // 4. peel vertices with out-degree less than k
                    for (::uint32_t k = 0; k < buckets.size(); ++k) {
                        if(buckets[k].empty()) continue;
                        while (!buckets[k].empty()) {
                            uint32_t vid = *buckets[k].begin();
                            buckets[k].erase(buckets[k].begin());
                            sub_out_coreness[vid] = k;
                            for (const ArrayEntry &ae: k_adj_in[vid]) {   //for in neighbors, decrease their out-degree
                                ::uint32_t nbr = ae.vid;
                                if(!(deleted[ae.eid])){
                                    deleted[ae.eid] = true;
                                    if(deg_out[nbr] > k && deg_in[nbr] >= current_k) {
                                        buckets[deg_out[nbr]].erase(nbr);
                                        --deg_out[nbr];
                                        buckets[deg_out[nbr]].insert(nbr);
                                    }
                                }
                            }
                            for (const ArrayEntry &ae: k_adj_out[vid]) {  //for out neighbors, check their in-degree to see if it >= k
                                ::uint32_t nbr = ae.vid;
                                if (!(deleted[ae.eid])) {
                                    deleted[ae.eid] = true;
                                    --deg_in[nbr];
                                    if(deg_in[nbr] < current_k){
                                        //ASSERT_MSG(deg_out[nbr] >= k, "wrong out degree error " << k << " vid " << nbr << " " << deg_out[nbr] << " " << k);
                                        buckets[deg_out[nbr]].erase(nbr);
                                        buckets[k].insert(nbr);
                                    }
                                }
                            }
                        }
                    }
                    //record the (current_k, a_{u}^{current_k}) pair for all vertices
                    for (uint32_t vid = 0; vid < tmp_n_; ++vid) {
                        uint32_t u = node_map_reverse[vid];
                        l_max[u][current_k] = sub_out_coreness[vid];
                    }
                }
            }
            auto test2 = std::chrono::steady_clock::now();
            final_core_update = test2 - test1;
        }
        else{
            /*std::chrono::duration<double>*/double out_initialzation = 0, initialzation = 0, ed_calculation = 0, dfs = 0;
            std::chrono::duration<double> final_core_update;
            uint32_t dfs_search_space = 0;
            auto start = omp_get_wtime();

            vector<vector<vector<ArrayEntry>>> total_adj_in(M_ + 1), total_adj_out(M_ + 1); //all the adj list of (k,0)-cores with 0 <= k <= M
            vector<vector<uint32_t>> total_mED_out(n_), total_mPED_out(n_), total_in_degree(n_);
            #pragma omp parallel for num_threads(lmax_number_of_threads)
            for(int k = 0; k <= M_; ++k){
                total_adj_in[k].resize(n_);
                total_adj_out[k].resize(n_);
                total_mED_out[k].resize(n_, 0);
                total_in_degree[k].resize(n_, 0);
                for (uint32_t eid = 0; eid < edges_.size(); ++eid) {
                    const uint32_t v1 = edges_[eid].first;
                    const uint32_t v2 = edges_[eid].second;
                    if(k_max[v1] >= k && k_max[v2] >= k){
                        total_adj_in[k][v2].push_back({v1, eid});
                        total_adj_out[k][v1].push_back({v2, eid});
                        if (l_max[v2][k] >= l_max[v1][k]) {
                            ++total_mED_out[k][v1];
                        }
                        ++total_in_degree[k][v2];
                    }
                }
            }

            /*original version, direct parallel*/
            //#pragma omp parallel for num_threads(lmax_number_of_threads)
            for(int k = 0; k <= M_; ++k){
                /*construct adj list in the (k,0)-core*/
                //auto test1 = std::chrono::steady_clock::now();
                auto test1 = omp_get_wtime();
                vector<vector<ArrayEntry>> k_adj_in(n_), k_adj_out(n_);
                vector<uint32_t> mED_out(n_, 0), mPED_out(n_, 0), in_degree(n_, 0);
                k_adj_in = total_adj_in[k];
                k_adj_out = total_adj_out[k];
                mED_out = total_mED_out[k];
                in_degree = total_in_degree[k];
//                for (uint32_t eid = 0; eid < edges_.size(); ++eid) {
//                    const uint32_t v1 = edges_[eid].first;
//                    const uint32_t v2 = edges_[eid].second;
//                    if(k_max[v1] >= k && k_max[v2] >= k){
//                        k_adj_in[v2].push_back({v1, eid});
//                        k_adj_out[v1].push_back({v2, eid});
//                        if (l_max[v2][k] >= l_max[v1][k]) {
//                            mED_out[v1]++;
//                        }
//                        ++in_degree[v2];
//                    }
//                }

                //auto test2 = std::chrono::steady_clock::now();
                auto test2 = omp_get_wtime();

                /*use the DFS-based method to maintain l_{max}(v,k) value of vertices*/
                uint32_t root = modified_edges[0].first;
                if (l_max[modified_edges[0].second][k] < l_max[modified_edges[0].first][k]) {
                    root = modified_edges[0].second;
                }
                uint32_t k_M_ = l_max[root][k];
                list<uint32_t> lsQ;
                vector<bool> vbVisited(n_, false);
                vector<bool> vbDeleted(n_, false);
                vector<uint32_t> m_d(n_, 0);    //candidate degree of vertices


                /*calculate PED value of vertices*/
                for (uint32_t vid = 0; vid < n_; ++vid) {
                    if(!k_adj_out.empty()){
                        for (auto neighbors: k_adj_out[vid]) {
                            if (l_max[neighbors.vid][k] > l_max[vid][k] ||
                                (l_max[neighbors.vid][k] == l_max[vid][k] && mED_out[neighbors.vid] > l_max[vid][k])) {
                                mPED_out[vid]++;
                            }
                        }
                    }
                }
                total_mPED_out[k] = mPED_out;

                //auto test3 = std::chrono::steady_clock::now();
                auto test3 = omp_get_wtime();
                m_d[root] = mPED_out[root], vbVisited[root] = true, lsQ.push_back(root);
                while (!lsQ.empty()) {
                    uint32_t cur_node = lsQ.front();
                    lsQ.pop_front();
                    if (m_d[cur_node] > k_M_) {
                        for (auto in_neighbor: k_adj_in[cur_node]) {
                            ++dfs_search_space;
                            if (l_max[in_neighbor.vid][k] == k_M_
                                && !vbVisited[in_neighbor.vid]
                                && mED_out[in_neighbor.vid] > k_M_){
                                lsQ.push_back(in_neighbor.vid);
                                m_d[in_neighbor.vid] = m_d[in_neighbor.vid] + mPED_out[in_neighbor.vid];
                                vbVisited[in_neighbor.vid] = true;
                            }
                        }
                    } else {
                        if (!vbDeleted[cur_node]) {
                            lMaxRemove(m_d, in_degree, vbDeleted, cur_node, k_adj_in, k_adj_out, k, k_M_);
                        }
                    }
                }
                for (uint32_t vid = 0; vid < n_; ++vid) {
                    if (vbVisited[vid] && !vbDeleted[vid]) {
                        ++l_max[vid][k];
                    }
                }
                //auto test4 = std::chrono::steady_clock::now();
                auto test4 = omp_get_wtime();

                initialzation += test2 - test1;
                ed_calculation += test3 - test2;
                dfs += test4 - test3;
            }

            auto end = omp_get_wtime();
            printf("dfs_search_space: %u\n", dfs_search_space);
            auto test1 = std::chrono::steady_clock::now();

            if(use_hindex){
                //using parallel-h-index based method to update the l_{max} value of (M_,0)-core
                for(uint32_t current_k = M_ + 1 ; current_k <= M_ + 1; ++current_k){
                    vector<vector<ArrayEntry>> k_adj_in(n_), k_adj_out(n_);
                    vector<pair<uint32_t,uint32_t>> cur_k_list;
                    vector<bool> is_in_subgraph (n_, false);   //wether the vertex is in the current (M_,0)-core
                    vector<uint32_t> sub_out_coreness(n_, 0);  //the l_{max} value of vertices in the current (M_,0)-core
                    for (uint32_t eid = 0; eid < edges_.size(); ++eid) {
                        const uint32_t v1 = edges_[eid].first;
                        const uint32_t v2 = edges_[eid].second;
                        if(k_max[v1] >= current_k && k_max[v2] >= current_k){
                            k_adj_in[v2].push_back({v1, eid});
                            k_adj_out[v1].push_back({v2, eid});
                            cur_k_list.push_back(edges_[eid]);
                        }
                    }
                    for(uint32_t vid = 0; vid < n_; ++vid){
                        if(k_max[vid] >= current_k){
                            is_in_subgraph[vid] = true;
                            if(l_max[vid][current_k] > sub_out_coreness[vid]){
                                sub_out_coreness[vid] = l_max[vid][current_k];
                            }
                            else{
                                sub_out_coreness[vid] = k_adj_out[vid].size();
                            }
                        }
                    }
                    //parallel h-index based method
                    bool flag = true;
                    uint32_t round_cnt = 0;
                    while (flag){
                        flag = false;
                        #pragma omp parallel for num_threads(lmax_number_of_threads)
                        for(uint32_t vid = 0; vid < n_; ++vid){
                            if(is_in_subgraph[vid]){
                                vector<uint32_t> tmp_neighbor_out_coreness(k_adj_out[vid].size(),0);
                                for(uint32_t i = 0; i < k_adj_out[vid].size(); ++i){
                                    tmp_neighbor_out_coreness[i] = sub_out_coreness[k_adj_out[vid][i].vid];
                                }
                                uint32_t tmp_h_index = hIndex(tmp_neighbor_out_coreness);
                                if(tmp_h_index < sub_out_coreness[vid]){
                                    sub_out_coreness[vid] = tmp_h_index;
                                    flag = true;
                                }
                            }
                        }
                        round_cnt++;
                    }
                    //cout << "insert round_cnt: " << round_cnt << endl;
                    for(uint32_t vid = 0; vid < n_; ++vid){
                        if(is_in_subgraph[vid]){
                            l_max[vid][current_k] = sub_out_coreness[vid];
                        }
                    }
                }
            }
            if(!use_hindex){
                //using peeling method to update the l_{max} value of (M_+1,0)-core
                // initialize adjacency arrays for the current k-list
                for(uint32_t current_k = M_ + 1 ; current_k <= M_ + 1 ; ++current_k){
                    vector<vector<ArrayEntry>> k_adj_in, k_adj_out;
                    // 1 map the node id in k-list -> consistent, zero-based node id
                    set<::uint32_t> node_recorder;
                    map<::uint32_t,uint32_t> node_map, node_map_reverse;
                    vector<pair<uint32_t,uint32_t>> cur_k_list;
                    ::uint32_t tmp_n_ = 0;       //the number of vertices in the current k-list
                    ::uint32_t tmp_m_ = 0;          //the number of edges in the current k-list
                    for (uint32_t eid = 0; eid < edges_.size(); ++eid) {
                        const uint32_t v1 = edges_[eid].first;
                        const uint32_t v2 = edges_[eid].second;
                        if(k_max[v1] >= current_k && k_max[v2] >= current_k){
                            ++tmp_m_;
                            cur_k_list.push_back(edges_[eid]);
                            node_recorder.insert(v1);
                            node_recorder.insert(v2);
                        }
                    }
                    for (uint32_t vid : node_recorder) {
                        node_map[vid] = std::distance(node_recorder.begin(), node_recorder.find(vid));
                        node_map_reverse[node_map[vid]] = vid;    //map the consistent, zero-based node id -> node id in k-list
                    }
                    tmp_n_ = node_recorder.size();       //the number of vertices in the current k-list
                    k_adj_in.resize(tmp_n_);
                    k_adj_out.resize(tmp_n_);
                    // 2 initialize the adjacency arrays
                    for (uint32_t eid = 0; eid < cur_k_list.size(); ++eid) {
                        const uint32_t v1 = cur_k_list[eid].first;
                        const uint32_t v2 = cur_k_list[eid].second;
                        k_adj_out[node_map[v1]].push_back({node_map[v2], eid });
                        k_adj_in[node_map[v2]].push_back({node_map[v1], eid});
                    }
                    for (uint32_t vid = 0; vid < tmp_n_; ++vid) {
                        std::sort(k_adj_out[vid].begin(), k_adj_out[vid].end(),
                                  [](const ArrayEntry& ae1, const ArrayEntry& ae2) {
                                      return ae1.vid < ae2.vid;
                                  });
                        std::sort(k_adj_in[vid].begin(), k_adj_in[vid].end(),
                                  [](const ArrayEntry& ae1, const ArrayEntry& ae2) {
                                      return ae1.vid < ae2.vid;
                                  });
                    }
                    // bin-sort peeling for k-list w.r.t the out-degree, get [(k,l)-core] for all the l values
                    // note that when we delete v, for its in-neighbor w, we maintain their sub_out_coreness(a_{w}^{k})
                    // for its out-neighbor u, we maintain their in_degree in current k-list, if the in_degree < k after v's deletion
                    // then we have a_{v}^{k} = a_{u}^{k}, and we delete u as well.

                    // 1.initialization
                    std::vector<::uint32_t> deg_in(tmp_n_, 0), deg_out(tmp_n_, 0);
                    std::vector<::uint32_t> sub_out_coreness(tmp_n_, 0);
                    std::vector<bool> deleted(tmp_m_, false);  //the edge is deleted or not?
                    // 2. get the in-degree
                    ::uint32_t max_out_deg = 0;
                    for (uint32_t vid = 0; vid < tmp_n_; ++vid) {
                        deg_out[vid] = k_adj_out[vid].size();
                        max_out_deg = std::max(max_out_deg, deg_out[vid]);
                        deg_in[vid] = k_adj_in[vid].size();
                    }

                    // 3. initialize buckets
                    vector<unordered_set<::uint32_t>> buckets(max_out_deg + 1);
                    for (uint32_t vid = 0; vid < tmp_n_; ++vid) {
                        ASSERT_MSG(deg_out[vid] <= max_out_deg, "wrong sub-out-degree error " << current_k << " vid " << vid << " " << deg_out[vid] << " " << max_out_deg);
                        buckets[deg_out[vid]].insert(vid);
                    }

                    // 4. peel vertices with out-degree less than k
                    for (::uint32_t k = 0; k < buckets.size(); ++k) {
                        if(buckets[k].empty()) continue;
                        while (!buckets[k].empty()) {
                            uint32_t vid = *buckets[k].begin();
                            buckets[k].erase(buckets[k].begin());
                            sub_out_coreness[vid] = k;
                            for (const ArrayEntry &ae: k_adj_in[vid]) {   //for in neighbors, decrease their out-degree
                                ::uint32_t nbr = ae.vid;
                                if(!(deleted[ae.eid])){
                                    deleted[ae.eid] = true;
                                    if(deg_out[nbr] > k && deg_in[nbr] >= current_k) {
                                        buckets[deg_out[nbr]].erase(nbr);
                                        --deg_out[nbr];
                                        buckets[deg_out[nbr]].insert(nbr);
                                    }
                                }
                            }
                            for (const ArrayEntry &ae: k_adj_out[vid]) {  //for out neighbors, check their in-degree to see if it >= k
                                ::uint32_t nbr = ae.vid;
                                if (!(deleted[ae.eid])) {
                                    deleted[ae.eid] = true;
                                    --deg_in[nbr];
                                    if(deg_in[nbr] < current_k){
                                        //ASSERT_MSG(deg_out[nbr] >= k, "wrong out degree error " << k << " vid " << nbr << " " << deg_out[nbr] << " " << k);
                                        buckets[deg_out[nbr]].erase(nbr);
                                        buckets[k].insert(nbr);
                                    }
                                }
                            }
                        }
                    }
                    //record the (current_k, a_{u}^{current_k}) pair for all vertices
                    for (uint32_t vid = 0; vid < tmp_n_; ++vid) {
                        uint32_t u = node_map_reverse[vid];
                        l_max[u][current_k] = sub_out_coreness[vid];
                    }
                }
            }

            auto test2 = std::chrono::steady_clock::now();
            final_core_update = test2 - test1;
        }
    }
    else{
        //for all the (k,0)-cores with 0 <= k <= N, we maintain the l_{max}(v, k) value of vertices
        //in the DFS search based way given edge deletion.
        if(k0core_pruning && reuse_pruning){
            uint32_t dfs_search_space = 0;
            vector<vector<vector<ArrayEntry>>> total_adj_in(M_ + 1), total_adj_out(M_ + 1); //all the adj list of (k,0)-cores with 0 <= k <= M
            vector<vector<uint32_t>> total_mED_out(n_), total_mPED_out(n_), total_in_degree(n_);

            //first get the unique (k, 0)-cores
            set<uint32_t, greater<uint32_t >> k0_value_set;
            vector<uint32_t> unique_k0_cores;
            for (uint32_t i = 0; i < n_; ++i) {
                if(k_max[i] <= M_){
                    k0_value_set.insert(k_max[i]);
                }
            }
            for(uint32_t i : k0_value_set){
                unique_k0_cores.push_back(i);
            }
           if(print_kocore_efficiency)  printf("k0core pruning efficiency: %f, %d, %d\n", (float)k0_value_set.size() / (float)(M_ + 1), k0_value_set.size(), M_ + 1);

            //do initialzation for reuse_pruning
            //ASSERT_MSG(lmax_number_of_threads <= M_ + 1, "reuse pruning error, the number of threads is larger than the number of (k,0)-cores " << lmax_number_of_threads << " " << M_ + 1);
            int small_batch_size = lmax_number_of_threads;
            int small_batch_number = (M_ + 1) / lmax_number_of_threads;
            int number_gap_in_batch = small_batch_number;
            if((M_ + 1) % lmax_number_of_threads != 0){  //M_ + 1 is not divisible by lmax_number_of_threads
                ++small_batch_number;
            }
            vector<vector<int>> initialization_batch( small_batch_number);
            for (int i = 0; i < lmax_number_of_threads; ++i) {
                for(int k = small_batch_number - 2; k >=0 ; --k){
                    if(k0_value_set.find(i * number_gap_in_batch + k) != k0_value_set.end()){
                        initialization_batch[k].push_back(i * number_gap_in_batch + k);
                    }
                    //initialization_batch[k].push_back(i * number_gap_in_batch + k);
                }
                if((M_ + 1) % lmax_number_of_threads == 0 ){
                    if(k0_value_set.find(i * number_gap_in_batch + small_batch_number - 1) != k0_value_set.end()){
                        initialization_batch[small_batch_number - 1].push_back(i * number_gap_in_batch + small_batch_number - 1);
                    }
                    //initialization_batch[small_batch_number - 1].push_back(i * number_gap_in_batch + small_batch_number - 1);
                }
                if( i < (M_ + 1) % lmax_number_of_threads){
                    if(k0_value_set.find((small_batch_number - 1) * lmax_number_of_threads + i) != k0_value_set.end()){
                        initialization_batch[small_batch_number - 1].push_back((small_batch_number - 1) * lmax_number_of_threads + i);
                    }
                    //initialization_batch[small_batch_number - 1].push_back((small_batch_number - 1) * lmax_number_of_threads + i);
                }

            }

            // for(int k = 0 ; k < small_batch_number; ++k){
            //     for(int i = 0; i < initialization_batch[k].size(); ++i){
            //         printf("k: %d, batch[i]: %d \n", k ,initialization_batch[k][i]);
            //     }
            // }

            for(int b = small_batch_number - 1; b >= 0; --b){
                if(b == small_batch_number - 1){
                    #pragma omp parallel for num_threads(lmax_number_of_threads)
                    for(int i = 0; i < initialization_batch[b].size(); ++i){
                        int k = initialization_batch[b][i];
                        total_adj_in[k].resize(n_);
                        total_adj_out[k].resize(n_);
                        total_mED_out[k].resize(n_, 0);
                        total_in_degree[k].resize(n_, 0);
                        for (uint32_t eid = 0; eid < edges_.size(); ++eid) {
                            const uint32_t v1 = edges_[eid].first;
                            const uint32_t v2 = edges_[eid].second;
                            if(k_max[v1] >= k && k_max[v2] >= k){
                                total_adj_in[k][v2].push_back({v1, eid});
                                total_adj_out[k][v1].push_back({v2, eid});
                                if (l_max[v2][k] >= l_max[v1][k]) {
                                    ++total_mED_out[k][v1];
                                }
                                ++total_in_degree[k][v2];
                            }
                        }
                    }

                }
                else if(b == small_batch_number - 2){
                    #pragma omp parallel for num_threads(lmax_number_of_threads)
                    for(int i = 0; i < initialization_batch[b].size(); ++i){
                        int k = initialization_batch[b][i];
                        if(find(initialization_batch.back().begin(), initialization_batch.back().end(), k+1) != initialization_batch.back().end()){
                            total_adj_in[k] = total_adj_in[k + 1];
                            total_adj_out[k] = total_adj_out[k + 1];
                            total_mED_out[k] = total_mED_out[k + 1];
                            total_in_degree[k] = total_in_degree[k + 1];
                            for (uint32_t eid = 0; eid < edges_.size(); ++eid) {
                                const uint32_t v1 = edges_[eid].first;
                                const uint32_t v2 = edges_[eid].second;
                                if(k_max[v1] >= k && k_max[v2] >= k){
                                    if(!(k_max[v1] >= k+1 && k_max[v2] >= k+1)){
                                        total_adj_in[k][v2].push_back({v1, eid});
                                        total_adj_out[k][v1].push_back({v2, eid});
                                        if (l_max[v2][k] >= l_max[v1][k]) {
                                            ++total_mED_out[k][v1];
                                        }
                                        ++total_in_degree[k][v2];
                                    }
                                }
                            }
                        }
                        else{
                            total_adj_in[k].resize(n_);
                            total_adj_out[k].resize(n_);
                            total_mED_out[k].resize(n_, 0);
                            total_in_degree[k].resize(n_, 0);
                            for (uint32_t eid = 0; eid < edges_.size(); ++eid) {
                                const uint32_t v1 = edges_[eid].first;
                                const uint32_t v2 = edges_[eid].second;
                                if(k_max[v1] >= k && k_max[v2] >= k){
                                    total_adj_in[k][v2].push_back({v1, eid});
                                    total_adj_out[k][v1].push_back({v2, eid});
                                    if (l_max[v2][k] >= l_max[v1][k]) {
                                        ++total_mED_out[k][v1];
                                    }
                                    ++total_in_degree[k][v2];
                                }
                            }
                        }
                    }
                }
                else{
                    #pragma omp parallel for num_threads(lmax_number_of_threads)
                    for(int i = 0; i < initialization_batch[b].size(); ++i){
                        int k = initialization_batch[b][i];
                        total_adj_in[k] = total_adj_in[k + 1];
                        total_adj_out[k] = total_adj_out[k + 1];
                        total_mED_out[k] = total_mED_out[k + 1];
                        total_in_degree[k] = total_in_degree[k + 1];
                        for (uint32_t eid = 0; eid < edges_.size(); ++eid) {
                            const uint32_t v1 = edges_[eid].first;
                            const uint32_t v2 = edges_[eid].second;
                            if(k_max[v1] >= k && k_max[v2] >= k){
                                if(!(k_max[v1] >= k+1 && k_max[v2] >= k+1)){
                                    total_adj_in[k][v2].push_back({v1, eid});
                                    total_adj_out[k][v1].push_back({v2, eid});
                                    if (l_max[v2][k] >= l_max[v1][k]) {
                                        ++total_mED_out[k][v1];
                                    }
                                    ++total_in_degree[k][v2];
                                }
                            }
                        }
                    }
                }
            }
            #pragma omp parallel for num_threads(lmax_number_of_threads)
            for(int k = M_ - 1; k >= 0; --k){
                if(k0_value_set.find(k) == k0_value_set.end()){
                    total_adj_in[k] = total_adj_in[k + 1];
                    total_adj_out[k] = total_adj_out[k + 1];
                    total_mED_out[k] = total_mED_out[k + 1];
                    total_in_degree[k] = total_in_degree[k + 1];
                }
            }


            #pragma omp parallel for num_threads(lmax_number_of_threads)
            for(uint32_t k : unique_k0_cores) {
                /*construct adj list in the (k,0)-core*/
                vector<vector<ArrayEntry>> k_adj_in, k_adj_out;
                k_adj_in.resize(n_), k_adj_out.resize(n_);
                for (uint32_t vid = 0; vid < n_; ++vid) {
                    if (k_max[vid] >= k) {
                        for (auto in_neighbor: adj_in[vid]) {
                            if (k_max[in_neighbor.vid] >= k) {
                                k_adj_in[vid].push_back({in_neighbor.vid, in_neighbor.eid});
                            }
                        }
                        for (auto out_neighbor: adj_out[vid]) {
                            if (k_max[out_neighbor.vid] >= k) {
                                k_adj_out[vid].push_back({out_neighbor.vid, out_neighbor.eid});
                            }
                        }
                    }
                }
                /*use the DFS-based method to maintain l_{max}(v,k) value of vertices*/
                uint32_t root = modified_edges[0].first;
                if (l_max[modified_edges[0].second][k] < l_max[modified_edges[0].first][k]) {
                    root = modified_edges[0].second;
                }
                uint32_t k_M_ = l_max[root][k];

                /*find in-core of deleted edges*/
                vector<uint32_t> m_d(n_, 0);    //candidate degree of vertices
                vector<vector<ArrayEntry>> sub_adj_in(n_), sub_adj_out(n_);  //record structure of out-core graph
                vector<bool> be_in_outcore(n_, false);
                if(l_max[modified_edges[0].second][k] == l_max[modified_edges[0].first][k]){
                    lMaxFindOutcore(modified_edges[0].first,  m_d, sub_adj_in, sub_adj_out, be_in_outcore,
                                    k, k_M_, k_adj_out);
                    lMaxFindOutcore(modified_edges[0].second , m_d, sub_adj_in, sub_adj_out, be_in_outcore,
                                    k, k_M_, k_adj_out);
                }
                else{
                    lMaxFindOutcore(root ,m_d, sub_adj_in, sub_adj_out, be_in_outcore,
                                    k, k_M_, k_adj_out);
                }

                /*evict disqualified vertices using bin-sort, similar to decomposition algo*/
                vector<unordered_set<uint32_t>> buckets(*max_element(m_d.begin(), m_d.end()) + 1);
                vector<bool> deleted(m_, false);
                for(uint32_t vid = 0; vid < n_; ++vid){
                    if(be_in_outcore[vid]){
                        ASSERT_MSG(m_d[vid] >= 0, "m_d[vid] is out of range: " << m_d[vid]);
                        buckets[m_d[vid]].insert(vid);
                    }
                }

                for (::uint32_t b = 0; b < buckets.size(); ++b) {
                    if(buckets[b].empty()){
                        continue;
                    }
                    if(b >= k_M_){
                        break;
                    }
                    for (auto vid: buckets[b]) {
                        if(be_in_outcore[vid]){
                            if(m_d[vid] < k_M_){
                                l_max[vid][k]--;
                                //dif_kmax_M_group.push_back(vid);
                                for (auto in_neighbor: sub_adj_in[vid]) {
                                    if(!deleted[in_neighbor.eid]){
                                        deleted[in_neighbor.eid] = true;
                                    }
                                    if(m_d[in_neighbor.vid] > m_d[vid]){
                                        buckets[m_d[in_neighbor.vid]].erase(in_neighbor.vid);
                                        m_d[in_neighbor.vid]--;
                                        buckets[m_d[in_neighbor.vid]].insert(in_neighbor.vid);
                                    }
                                }
                                for(auto out_neighbor : sub_adj_out[vid]){
                                    if(!deleted[out_neighbor.eid]){
                                        deleted[out_neighbor.eid] = true;
                                    }
                                }
                            } else{
                                break;
                            }
                        }
                    }
                }
            }
            //printf("dfs_search_space: %u\n", dfs_search_space);


            if(use_hindex){
                //using parallel-h-index based method to update the l_{max} value of (M_,0)-core
                for(uint32_t current_k = M_  ; current_k <= M_ ; ++current_k){
                    vector<vector<ArrayEntry>> k_adj_in(n_), k_adj_out(n_);
                    vector<pair<uint32_t,uint32_t>> cur_k_list;
                    vector<bool> is_in_subgraph (n_, false);   //wether the vertex is in the current (M_,0)-core
                    vector<uint32_t> sub_out_coreness(n_, 0);  //the l_{max} value of vertices in the current (M_,0)-core
                    for (uint32_t eid = 0; eid < edges_.size(); ++eid) {
                        const uint32_t v1 = edges_[eid].first;
                        const uint32_t v2 = edges_[eid].second;
                        if(k_max[v1] >= current_k && k_max[v2] >= current_k){
                            k_adj_in[v2].push_back({v1, eid});
                            k_adj_out[v1].push_back({v2, eid});
                            cur_k_list.push_back(edges_[eid]);
                        }
                    }
                    for(uint32_t vid = 0; vid < n_; ++vid){
                        if(k_max[vid] >= current_k){
                            is_in_subgraph[vid] = true;
                            if(l_max[vid][current_k] > sub_out_coreness[vid]){
                                sub_out_coreness[vid] = l_max[vid][current_k];
                            }
                            else{
                                sub_out_coreness[vid] = k_adj_out[vid].size();
                            }
                        }
                    }
                    //parallel h-index based method
                    bool flag = true;
                    while (flag){
                        flag = false;
                        #pragma omp parallel for num_threads(lmax_number_of_threads)
                        for(uint32_t vid = 0; vid < n_; ++vid){
                            if(is_in_subgraph[vid]){
                                vector<uint32_t> tmp_neighbor_out_coreness(k_adj_out[vid].size(),0);
                                for(uint32_t i = 0; i < k_adj_out[vid].size(); ++i){
                                    tmp_neighbor_out_coreness[i] = sub_out_coreness[k_adj_out[vid][i].vid];
                                }
                                if(hIndex(tmp_neighbor_out_coreness) < sub_out_coreness[vid]){
                                    sub_out_coreness[vid] = hIndex(tmp_neighbor_out_coreness);
                                    flag = true;
                                }
                            }
                        }
                    }
                    for(uint32_t vid = 0; vid < n_; ++vid){
                        if(is_in_subgraph[vid]){
                            l_max[vid][current_k] = sub_out_coreness[vid];
                        }
                    }

                }
            }
            if(!use_hindex){
                //using peeling method to update the l_{max} value of (M_,0)-core
                // initialize adjacency arrays for the current k-list
                for(uint32_t current_k = M_  ; current_k <= M_ ; ++current_k){
                    vector<vector<ArrayEntry>> k_adj_in, k_adj_out;
                    // 1 map the node id in k-list -> consistent, zero-based node id
                    set<::uint32_t> node_recorder;
                    map<::uint32_t,uint32_t> node_map, node_map_reverse;
                    vector<pair<uint32_t,uint32_t>> cur_k_list;
                    ::uint32_t tmp_n_ = 0;       //the number of vertices in the current k-list
                    ::uint32_t tmp_m_ = 0;          //the number of edges in the current k-list
                    for (uint32_t eid = 0; eid < edges_.size(); ++eid) {
                        const uint32_t v1 = edges_[eid].first;
                        const uint32_t v2 = edges_[eid].second;
                        if(k_max[v1] >= current_k && k_max[v2] >= current_k){
                            ++tmp_m_;
                            cur_k_list.push_back(edges_[eid]);
                            node_recorder.insert(v1);
                            node_recorder.insert(v2);
                        }
                    }
                    for (uint32_t vid : node_recorder) {
                        node_map[vid] = std::distance(node_recorder.begin(), node_recorder.find(vid));
                        node_map_reverse[node_map[vid]] = vid;    //map the consistent, zero-based node id -> node id in k-list
                    }
                    tmp_n_ = node_recorder.size();       //the number of vertices in the current k-list
                    k_adj_in.resize(tmp_n_);
                    k_adj_out.resize(tmp_n_);
                    // 2 initialize the adjacency arrays
                    for (uint32_t eid = 0; eid < cur_k_list.size(); ++eid) {
                        const uint32_t v1 = cur_k_list[eid].first;
                        const uint32_t v2 = cur_k_list[eid].second;
                        k_adj_out[node_map[v1]].push_back({node_map[v2], eid });
                        k_adj_in[node_map[v2]].push_back({node_map[v1], eid });
                    }
                    for (uint32_t vid = 0; vid < tmp_n_; ++vid) {
                        std::sort(k_adj_out[vid].begin(), k_adj_out[vid].end(),
                                  [](const ArrayEntry& ae1, const ArrayEntry& ae2) {
                                      return ae1.vid < ae2.vid;
                                  });
                        std::sort(k_adj_in[vid].begin(), k_adj_in[vid].end(),
                                  [](const ArrayEntry& ae1, const ArrayEntry& ae2) {
                                      return ae1.vid < ae2.vid;
                                  });
                    }
                    // bin-sort peeling for k-list w.r.t the out-degree, get [(k,l)-core] for all the l values
                    // note that when we delete v, for its in-neighbor w, we maintain their sub_out_coreness(a_{w}^{k})
                    // for its out-neighbor u, we maintain their in_degree in current k-list, if the in_degree < k after v's deletion
                    // then we have a_{v}^{k} = a_{u}^{k}, and we delete u as well.

                    // 1.initialization
                    std::vector<::uint32_t> deg_in(tmp_n_, 0), deg_out(tmp_n_, 0);
                    std::vector<::uint32_t> sub_out_coreness(tmp_n_, 0);
                    std::vector<bool> deleted(tmp_m_, false);  //the edge is deleted or not?
                    // 2. get the in-degree
                    ::uint32_t max_out_deg = 0;
                    for (uint32_t vid = 0; vid < tmp_n_; ++vid) {
                        deg_out[vid] = k_adj_out[vid].size();
                        max_out_deg = std::max(max_out_deg, deg_out[vid]);
                        deg_in[vid] = k_adj_in[vid].size();
                    }

                    // 3. initialize buckets
                    vector<unordered_set<::uint32_t>> buckets(max_out_deg + 1);
                    for (uint32_t vid = 0; vid < tmp_n_; ++vid) {
                        ASSERT_MSG(deg_out[vid] <= max_out_deg, "wrong sub-out-degree error " << current_k << " vid " << vid << " " << deg_out[vid] << " " << max_out_deg);
                        buckets[deg_out[vid]].insert(vid);
                    }

                    // 4. peel vertices with out-degree less than k
                    for (::uint32_t k = 0; k < buckets.size(); ++k) {
                        if(buckets[k].empty()) continue;
                        while (!buckets[k].empty()) {
                            uint32_t vid = *buckets[k].begin();
                            buckets[k].erase(buckets[k].begin());
                            sub_out_coreness[vid] = k;
                            for (const ArrayEntry &ae: k_adj_in[vid]) {   //for in neighbors, decrease their out-degree
                                ::uint32_t nbr = ae.vid;
                                if(!(deleted[ae.eid])){
                                    deleted[ae.eid] = true;
                                    if(deg_out[nbr] > k && deg_in[nbr] >= current_k) {
                                        buckets[deg_out[nbr]].erase(nbr);
                                        --deg_out[nbr];
                                        buckets[deg_out[nbr]].insert(nbr);
                                    }
                                }
                            }
                            for (const ArrayEntry &ae: k_adj_out[vid]) {  //for out neighbors, check their in-degree to see if it >= k
                                ::uint32_t nbr = ae.vid;
                                if (!(deleted[ae.eid])) {
                                    deleted[ae.eid] = true;
                                    --deg_in[nbr];
                                    if(deg_in[nbr] < current_k){
                                        ASSERT_MSG(deg_out[nbr] >= k, "wrong out degree error " << k << " vid " << nbr << " " << deg_out[nbr] << " " << k);
                                        buckets[deg_out[nbr]].erase(nbr);
                                        buckets[k].insert(nbr);
                                    }
                                }
                            }
                        }
                    }
                    //record the (current_k, a_{u}^{current_k}) pair for all vertices
                    for (uint32_t vid = 0; vid < tmp_n_; ++vid) {
                        uint32_t u = node_map_reverse[vid];
                        l_max[u][current_k] = sub_out_coreness[vid];
                    }
                }
            }


            //update the l_{max} value of (k,0)-cores with identical structure
            if(unique_k0_cores.size() != M_){
                for(uint32_t k = unique_k0_cores.size() - 1; k > 0 ; --k){
                    if(unique_k0_cores[k] > unique_k0_cores[k-1] + 1){
                        for(uint32_t j = unique_k0_cores[k]; j > unique_k0_cores[k-1]; --j){
                            for(uint32_t vid = 0; vid < n_; ++vid){
                                if(k_max[vid] >= unique_k0_cores[k]){
                                    l_max[vid][j] = l_max[vid][unique_k0_cores[k]];
                                }
                            }
                        }
                    }
                }
            }


        }
        else if(k0core_pruning){
                        //first get the unique (k, 0)-cores
            vector<uint32_t> unique_k0_cores;
            set<uint32_t> k0_value_set;
            for (uint32_t i = 0; i < n_; ++i) {
                if(k_max[i] <= M_){
                    k0_value_set.insert(k_max[i]);
                }
            }
            for(uint32_t i : k0_value_set){
                unique_k0_cores.push_back(i);
            }
            //printf("k0core pruning efficiency: %f, %d, %d\n", (float)unique_k0_cores.size() / (float)(N_ + 1), unique_k0_cores.size(), N_ + 1);

            #pragma omp parallel for num_threads(lmax_number_of_threads)
            for(uint32_t k : unique_k0_cores) {
                /*construct adj list in the (k,0)-core*/
                vector<vector<ArrayEntry>> k_adj_in, k_adj_out;
                k_adj_in.resize(n_), k_adj_out.resize(n_);
                for (uint32_t vid = 0; vid < n_; ++vid) {
                    if (k_max[vid] >= k) {
                        for (auto in_neighbor: adj_in[vid]) {
                            if (k_max[in_neighbor.vid] >= k) {
                                k_adj_in[vid].push_back({in_neighbor.vid, in_neighbor.eid});
                            }
                        }
                        for (auto out_neighbor: adj_out[vid]) {
                            if (k_max[out_neighbor.vid] >= k) {
                                k_adj_out[vid].push_back({out_neighbor.vid, out_neighbor.eid});
                            }
                        }
                    }
                }
                /*use the DFS-based method to maintain l_{max}(v,k) value of vertices*/
                uint32_t root = modified_edges[0].first;
                if (l_max[modified_edges[0].second][k] < l_max[modified_edges[0].first][k]) {
                    root = modified_edges[0].second;
                }
                uint32_t k_M_ = l_max[root][k];

                /*find in-core of deleted edges*/
                vector<uint32_t> m_d(n_, 0);    //candidate degree of vertices
                vector<vector<ArrayEntry>> sub_adj_in(n_), sub_adj_out(n_);  //record structure of out-core graph
                vector<bool> be_in_outcore(n_, false);
                if(l_max[modified_edges[0].second][k] == l_max[modified_edges[0].first][k]){
                    lMaxFindOutcore(modified_edges[0].first,  m_d, sub_adj_in, sub_adj_out, be_in_outcore,
                                    k, k_M_, k_adj_out);
                    lMaxFindOutcore(modified_edges[0].second , m_d, sub_adj_in, sub_adj_out, be_in_outcore,
                                    k, k_M_, k_adj_out);
                }
                else{
                    lMaxFindOutcore(root ,m_d, sub_adj_in, sub_adj_out, be_in_outcore,
                                    k, k_M_, k_adj_out);
                }

                /*evict disqualified vertices using bin-sort, similar to decomposition algo*/
                vector<unordered_set<uint32_t>> buckets(*max_element(m_d.begin(), m_d.end()) + 1);
                vector<bool> deleted(m_, false);
                for(uint32_t vid = 0; vid < n_; ++vid){
                    if(be_in_outcore[vid]){
                        ASSERT_MSG(m_d[vid] >= 0, "m_d[vid] is out of range: " << m_d[vid]);
                        buckets[m_d[vid]].insert(vid);
                    }
                }

                for (::uint32_t b = 0; b < buckets.size(); ++b) {
                    if(buckets[b].empty()){
                        continue;
                    }
                    if(b >= k_M_){
                        break;
                    }
                    for (auto vid: buckets[b]) {
                        if(be_in_outcore[vid]){
                            if(m_d[vid] < k_M_){
                                l_max[vid][k]--;
                                //dif_kmax_M_group.push_back(vid);
                                for (auto in_neighbor: sub_adj_in[vid]) {
                                    if(!deleted[in_neighbor.eid]){
                                        deleted[in_neighbor.eid] = true;
                                    }
                                    if(m_d[in_neighbor.vid] > m_d[vid]){
                                        buckets[m_d[in_neighbor.vid]].erase(in_neighbor.vid);
                                        m_d[in_neighbor.vid]--;
                                        buckets[m_d[in_neighbor.vid]].insert(in_neighbor.vid);
                                    }
                                }
                                for(auto out_neighbor : sub_adj_out[vid]){
                                    if(!deleted[out_neighbor.eid]){
                                        deleted[out_neighbor.eid] = true;
                                    }
                                }
                            } else{
                                break;
                            }
                        }
                    }
                }
            }

            if(use_hindex){
                //using parallel-h-index based method to update the l_{max} value of (M_,0)-core
                for(uint32_t current_k = M_  ; current_k <= M_ ; ++current_k){
                    vector<vector<ArrayEntry>> k_adj_in(n_), k_adj_out(n_);
                    vector<pair<uint32_t,uint32_t>> cur_k_list;
                    vector<bool> is_in_subgraph (n_, false);   //wether the vertex is in the current (M_,0)-core
                    vector<uint32_t> sub_out_coreness(n_, 0);  //the l_{max} value of vertices in the current (M_,0)-core
                    for (uint32_t eid = 0; eid < edges_.size(); ++eid) {
                        const uint32_t v1 = edges_[eid].first;
                        const uint32_t v2 = edges_[eid].second;
                        if(k_max[v1] >= current_k && k_max[v2] >= current_k){
                            k_adj_in[v2].push_back({v1, eid});
                            k_adj_out[v1].push_back({v2, eid});
                            cur_k_list.push_back(edges_[eid]);
                        }
                    }
                    for(uint32_t vid = 0; vid < n_; ++vid){
                        if(k_max[vid] >= current_k){
                            is_in_subgraph[vid] = true;
                            if(l_max[vid][current_k] > sub_out_coreness[vid]){
                                sub_out_coreness[vid] = l_max[vid][current_k];
                            }
                            else{
                                sub_out_coreness[vid] = k_adj_out[vid].size();
                            }
                        }
                    }
                    //parallel h-index based method
                    bool flag = true;
                    while (flag){
                        flag = false;
                        #pragma omp parallel for num_threads(lmax_number_of_threads)
                        for(uint32_t vid = 0; vid < n_; ++vid){
                            if(is_in_subgraph[vid]){
                                vector<uint32_t> tmp_neighbor_out_coreness(k_adj_out[vid].size(),0);
                                for(uint32_t i = 0; i < k_adj_out[vid].size(); ++i){
                                    tmp_neighbor_out_coreness[i] = sub_out_coreness[k_adj_out[vid][i].vid];
                                }
                                if(hIndex(tmp_neighbor_out_coreness) < sub_out_coreness[vid]){
                                    sub_out_coreness[vid] = hIndex(tmp_neighbor_out_coreness);
                                    flag = true;
                                }
                            }
                        }
                    }
                    for(uint32_t vid = 0; vid < n_; ++vid){
                        if(is_in_subgraph[vid]){
                            l_max[vid][current_k] = sub_out_coreness[vid];
                        }
                    }

                }
            }
            if(!use_hindex){
                //using peeling method to update the l_{max} value of (M_,0)-core
                // initialize adjacency arrays for the current k-list
                for(uint32_t current_k = M_  ; current_k <= M_ ; ++current_k){
                    vector<vector<ArrayEntry>> k_adj_in, k_adj_out;
                    // 1 map the node id in k-list -> consistent, zero-based node id
                    set<::uint32_t> node_recorder;
                    map<::uint32_t,uint32_t> node_map, node_map_reverse;
                    vector<pair<uint32_t,uint32_t>> cur_k_list;
                    ::uint32_t tmp_n_ = 0;       //the number of vertices in the current k-list
                    ::uint32_t tmp_m_ = 0;          //the number of edges in the current k-list
                    for (uint32_t eid = 0; eid < edges_.size(); ++eid) {
                        const uint32_t v1 = edges_[eid].first;
                        const uint32_t v2 = edges_[eid].second;
                        if(k_max[v1] >= current_k && k_max[v2] >= current_k){
                            ++tmp_m_;
                            cur_k_list.push_back(edges_[eid]);
                            node_recorder.insert(v1);
                            node_recorder.insert(v2);
                        }
                    }
                    for (uint32_t vid : node_recorder) {
                        node_map[vid] = std::distance(node_recorder.begin(), node_recorder.find(vid));
                        node_map_reverse[node_map[vid]] = vid;    //map the consistent, zero-based node id -> node id in k-list
                    }
                    tmp_n_ = node_recorder.size();       //the number of vertices in the current k-list
                    k_adj_in.resize(tmp_n_);
                    k_adj_out.resize(tmp_n_);
                    // 2 initialize the adjacency arrays
                    for (uint32_t eid = 0; eid < cur_k_list.size(); ++eid) {
                        const uint32_t v1 = cur_k_list[eid].first;
                        const uint32_t v2 = cur_k_list[eid].second;
                        k_adj_out[node_map[v1]].push_back({node_map[v2], eid });
                        k_adj_in[node_map[v2]].push_back({node_map[v1], eid });
                    }
                    for (uint32_t vid = 0; vid < tmp_n_; ++vid) {
                        std::sort(k_adj_out[vid].begin(), k_adj_out[vid].end(),
                                  [](const ArrayEntry& ae1, const ArrayEntry& ae2) {
                                      return ae1.vid < ae2.vid;
                                  });
                        std::sort(k_adj_in[vid].begin(), k_adj_in[vid].end(),
                                  [](const ArrayEntry& ae1, const ArrayEntry& ae2) {
                                      return ae1.vid < ae2.vid;
                                  });
                    }
                    // bin-sort peeling for k-list w.r.t the out-degree, get [(k,l)-core] for all the l values
                    // note that when we delete v, for its in-neighbor w, we maintain their sub_out_coreness(a_{w}^{k})
                    // for its out-neighbor u, we maintain their in_degree in current k-list, if the in_degree < k after v's deletion
                    // then we have a_{v}^{k} = a_{u}^{k}, and we delete u as well.

                    // 1.initialization
                    std::vector<::uint32_t> deg_in(tmp_n_, 0), deg_out(tmp_n_, 0);
                    std::vector<::uint32_t> sub_out_coreness(tmp_n_, 0);
                    std::vector<bool> deleted(tmp_m_, false);  //the edge is deleted or not?
                    // 2. get the in-degree
                    ::uint32_t max_out_deg = 0;
                    for (uint32_t vid = 0; vid < tmp_n_; ++vid) {
                        deg_out[vid] = k_adj_out[vid].size();
                        max_out_deg = std::max(max_out_deg, deg_out[vid]);
                        deg_in[vid] = k_adj_in[vid].size();
                    }

                    // 3. initialize buckets
                    vector<unordered_set<::uint32_t>> buckets(max_out_deg + 1);
                    for (uint32_t vid = 0; vid < tmp_n_; ++vid) {
                        ASSERT_MSG(deg_out[vid] <= max_out_deg, "wrong sub-out-degree error " << current_k << " vid " << vid << " " << deg_out[vid] << " " << max_out_deg);
                        buckets[deg_out[vid]].insert(vid);
                    }

                    // 4. peel vertices with out-degree less than k
                    for (::uint32_t k = 0; k < buckets.size(); ++k) {
                        if(buckets[k].empty()) continue;
                        while (!buckets[k].empty()) {
                            uint32_t vid = *buckets[k].begin();
                            buckets[k].erase(buckets[k].begin());
                            sub_out_coreness[vid] = k;
                            for (const ArrayEntry &ae: k_adj_in[vid]) {   //for in neighbors, decrease their out-degree
                                ::uint32_t nbr = ae.vid;
                                if(!(deleted[ae.eid])){
                                    deleted[ae.eid] = true;
                                    if(deg_out[nbr] > k && deg_in[nbr] >= current_k) {
                                        buckets[deg_out[nbr]].erase(nbr);
                                        --deg_out[nbr];
                                        buckets[deg_out[nbr]].insert(nbr);
                                    }
                                }
                            }
                            for (const ArrayEntry &ae: k_adj_out[vid]) {  //for out neighbors, check their in-degree to see if it >= k
                                ::uint32_t nbr = ae.vid;
                                if (!(deleted[ae.eid])) {
                                    deleted[ae.eid] = true;
                                    --deg_in[nbr];
                                    if(deg_in[nbr] < current_k){
                                        ASSERT_MSG(deg_out[nbr] >= k, "wrong out degree error " << k << " vid " << nbr << " " << deg_out[nbr] << " " << k);
                                        buckets[deg_out[nbr]].erase(nbr);
                                        buckets[k].insert(nbr);
                                    }
                                }
                            }
                        }
                    }
                    //record the (current_k, a_{u}^{current_k}) pair for all vertices
                    for (uint32_t vid = 0; vid < tmp_n_; ++vid) {
                        uint32_t u = node_map_reverse[vid];
                        l_max[u][current_k] = sub_out_coreness[vid];
                    }
                }
            }


            //update the l_{max} value of (k,0)-cores with identical structure
            if(unique_k0_cores.size() != M_){
                for(uint32_t k = unique_k0_cores.size() - 1; k > 0 ; --k){
                    if(unique_k0_cores[k] > unique_k0_cores[k-1] + 1){
                        for(uint32_t j = unique_k0_cores[k]; j > unique_k0_cores[k-1]; --j){
                            for(uint32_t vid = 0; vid < n_; ++vid){
                                if(k_max[vid] >= unique_k0_cores[k]){
                                    l_max[vid][j] = l_max[vid][unique_k0_cores[k]];
                                }
                            }
                        }
                    }
                }
            }
        }
        else if(reuse_pruning){
            uint32_t dfs_search_space = 0;
            vector<vector<vector<ArrayEntry>>> total_adj_in(M_ ), total_adj_out(M_ ); //all the adj list of (k,0)-cores with 0 <= k <= M
            vector<vector<uint32_t>> total_mED_out(n_), total_mPED_out(n_), total_in_degree(n_);

            //do initialzation for reuse_pruning
            //ASSERT_MSG(lmax_number_of_threads <= M_ + 1, "reuse pruning error, the number of threads is larger than the number of (k,0)-cores " << lmax_number_of_threads << " " << M_ + 1);
            int small_batch_size = lmax_number_of_threads;
            int small_batch_number = (M_ ) / lmax_number_of_threads;
            int number_gap_in_batch = small_batch_number;
            if((M_ ) % lmax_number_of_threads != 0){  //M_ + 1 is not divisible by lmax_number_of_threads
                ++small_batch_number;
            }
            vector<vector<int>> initialization_batch( small_batch_number);
             for (int i = 0; i < lmax_number_of_threads; ++i) {
                for(int k = small_batch_number - 2; k >=0 ; --k){
                    initialization_batch[k].push_back(i * number_gap_in_batch + k);
                }
                if((M_ ) % lmax_number_of_threads == 0 ){
                    initialization_batch[small_batch_number - 1].push_back(i * number_gap_in_batch + small_batch_number - 1);
                }
                if( i < (M_ ) % lmax_number_of_threads){
                    initialization_batch[small_batch_number - 1].push_back((small_batch_number - 1) * lmax_number_of_threads + i);
                }

            }


            for(int b = small_batch_number - 1; b >= 0; --b){
                if(b == small_batch_number - 1){
                    #pragma omp parallel for num_threads(lmax_number_of_threads)
                    for(int i = 0; i < initialization_batch[b].size(); ++i){
                        int k = initialization_batch[b][i];
                        total_adj_in[k].resize(n_);
                        total_adj_out[k].resize(n_);
                        total_mED_out[k].resize(n_, 0);
                        total_in_degree[k].resize(n_, 0);
                        for (uint32_t eid = 0; eid < edges_.size(); ++eid) {
                            const uint32_t v1 = edges_[eid].first;
                            const uint32_t v2 = edges_[eid].second;
                            if(k_max[v1] >= k && k_max[v2] >= k){
                                total_adj_in[k][v2].push_back({v1, eid});
                                total_adj_out[k][v1].push_back({v2, eid});
                                if (l_max[v2][k] >= l_max[v1][k]) {
                                    ++total_mED_out[k][v1];
                                }
                                ++total_in_degree[k][v2];
                            }
                        }
                    }

                }
                else if(b == small_batch_number - 2){
                    #pragma omp parallel for num_threads(lmax_number_of_threads)
                    for(int i = 0; i < initialization_batch[b].size(); ++i){
                        int k = initialization_batch[b][i];
                        if(find(initialization_batch.back().begin(), initialization_batch.back().end(), k+1) != initialization_batch.back().end()){
                            total_adj_in[k] = total_adj_in[k + 1];
                            total_adj_out[k] = total_adj_out[k + 1];
                            total_mED_out[k] = total_mED_out[k + 1];
                            total_in_degree[k] = total_in_degree[k + 1];
                            for (uint32_t eid = 0; eid < edges_.size(); ++eid) {
                                const uint32_t v1 = edges_[eid].first;
                                const uint32_t v2 = edges_[eid].second;
                                if(k_max[v1] >= k && k_max[v2] >= k){
                                    if(!(k_max[v1] >= k+1 && k_max[v2] >= k+1)){
                                        total_adj_in[k][v2].push_back({v1, eid});
                                        total_adj_out[k][v1].push_back({v2, eid});
                                        if (l_max[v2][k] >= l_max[v1][k]) {
                                            ++total_mED_out[k][v1];
                                        }
                                        ++total_in_degree[k][v2];
                                    }
                                }
                            }
                        }
                        else{
                            total_adj_in[k].resize(n_);
                            total_adj_out[k].resize(n_);
                            total_mED_out[k].resize(n_, 0);
                            total_in_degree[k].resize(n_, 0);
                            for (uint32_t eid = 0; eid < edges_.size(); ++eid) {
                                const uint32_t v1 = edges_[eid].first;
                                const uint32_t v2 = edges_[eid].second;
                                if(k_max[v1] >= k && k_max[v2] >= k){
                                    total_adj_in[k][v2].push_back({v1, eid});
                                    total_adj_out[k][v1].push_back({v2, eid});
                                    if (l_max[v2][k] >= l_max[v1][k]) {
                                        ++total_mED_out[k][v1];
                                    }
                                    ++total_in_degree[k][v2];
                                }
                            }
                        }
                    }
                }
                else{
                    #pragma omp parallel for num_threads(lmax_number_of_threads)
                    for(int i = 0; i < initialization_batch[b].size(); ++i){
                        int k = initialization_batch[b][i];
                        total_adj_in[k] = total_adj_in[k + 1];
                        total_adj_out[k] = total_adj_out[k + 1];
                        total_mED_out[k] = total_mED_out[k + 1];
                        total_in_degree[k] = total_in_degree[k + 1];
                        for (uint32_t eid = 0; eid < edges_.size(); ++eid) {
                            const uint32_t v1 = edges_[eid].first;
                            const uint32_t v2 = edges_[eid].second;
                            if(k_max[v1] >= k && k_max[v2] >= k){
                                if(!(k_max[v1] >= k+1 && k_max[v2] >= k+1)){
                                    total_adj_in[k][v2].push_back({v1, eid});
                                    total_adj_out[k][v1].push_back({v2, eid});
                                    if (l_max[v2][k] >= l_max[v1][k]) {
                                        ++total_mED_out[k][v1];
                                    }
                                    ++total_in_degree[k][v2];
                                }
                            }
                        }
                    }
                }
            }



            #pragma omp parallel for num_threads(lmax_number_of_threads)
            for(uint32_t k = 0; k < M_; ++k) {
                /*construct adj list in the (k,0)-core*/
                vector<vector<ArrayEntry>> k_adj_in, k_adj_out;
                k_adj_in.resize(n_), k_adj_out.resize(n_);
                for (uint32_t vid = 0; vid < n_; ++vid) {
                    if (k_max[vid] >= k) {
                        for (auto in_neighbor: adj_in[vid]) {
                            if (k_max[in_neighbor.vid] >= k) {
                                k_adj_in[vid].push_back({in_neighbor.vid, in_neighbor.eid});
                            }
                        }
                        for (auto out_neighbor: adj_out[vid]) {
                            if (k_max[out_neighbor.vid] >= k) {
                                k_adj_out[vid].push_back({out_neighbor.vid, out_neighbor.eid});
                            }
                        }
                    }
                }
                /*use the DFS-based method to maintain l_{max}(v,k) value of vertices*/
                uint32_t root = modified_edges[0].first;
                if (l_max[modified_edges[0].second][k] < l_max[modified_edges[0].first][k]) {
                    root = modified_edges[0].second;
                }
                uint32_t k_M_ = l_max[root][k];

                /*find in-core of deleted edges*/
                vector<uint32_t> m_d(n_, 0);    //candidate degree of vertices
                vector<vector<ArrayEntry>> sub_adj_in(n_), sub_adj_out(n_);  //record structure of out-core graph
                vector<bool> be_in_outcore(n_, false);
                if(l_max[modified_edges[0].second][k] == l_max[modified_edges[0].first][k]){
                    lMaxFindOutcore(modified_edges[0].first,  m_d, sub_adj_in, sub_adj_out, be_in_outcore,
                                    k, k_M_, k_adj_out);
                    lMaxFindOutcore(modified_edges[0].second , m_d, sub_adj_in, sub_adj_out, be_in_outcore,
                                    k, k_M_, k_adj_out);
                }
                else{
                    lMaxFindOutcore(root ,m_d, sub_adj_in, sub_adj_out, be_in_outcore,
                                    k, k_M_, k_adj_out);
                }

                /*evict disqualified vertices using bin-sort, similar to decomposition algo*/
                vector<unordered_set<uint32_t>> buckets(*max_element(m_d.begin(), m_d.end()) + 1);
                vector<bool> deleted(m_, false);
                for(uint32_t vid = 0; vid < n_; ++vid){
                    if(be_in_outcore[vid]){
                        ASSERT_MSG(m_d[vid] >= 0, "m_d[vid] is out of range: " << m_d[vid]);
                        buckets[m_d[vid]].insert(vid);
                    }
                }

                for (::uint32_t b = 0; b < buckets.size(); ++b) {
                    if(buckets[b].empty()){
                        continue;
                    }
                    if(b >= k_M_){
                        break;
                    }
                    for (auto vid: buckets[b]) {
                        if(be_in_outcore[vid]){
                            if(m_d[vid] < k_M_){
                                l_max[vid][k]--;
                                //dif_kmax_M_group.push_back(vid);
                                for (auto in_neighbor: sub_adj_in[vid]) {
                                    if(!deleted[in_neighbor.eid]){
                                        deleted[in_neighbor.eid] = true;
                                    }
                                    if(m_d[in_neighbor.vid] > m_d[vid]){
                                        buckets[m_d[in_neighbor.vid]].erase(in_neighbor.vid);
                                        m_d[in_neighbor.vid]--;
                                        buckets[m_d[in_neighbor.vid]].insert(in_neighbor.vid);
                                    }
                                }
                                for(auto out_neighbor : sub_adj_out[vid]){
                                    if(!deleted[out_neighbor.eid]){
                                        deleted[out_neighbor.eid] = true;
                                    }
                                }
                            } else{
                                break;
                            }
                        }
                    }
                }
            }
            //printf("dfs_search_space: %u\n", dfs_search_space);


            if(use_hindex){
                //using parallel-h-index based method to update the l_{max} value of (M_,0)-core
                for(uint32_t current_k = M_  ; current_k <= M_ ; ++current_k){
                    vector<vector<ArrayEntry>> k_adj_in(n_), k_adj_out(n_);
                    vector<pair<uint32_t,uint32_t>> cur_k_list;
                    vector<bool> is_in_subgraph (n_, false);   //wether the vertex is in the current (M_,0)-core
                    vector<uint32_t> sub_out_coreness(n_, 0);  //the l_{max} value of vertices in the current (M_,0)-core
                    for (uint32_t eid = 0; eid < edges_.size(); ++eid) {
                        const uint32_t v1 = edges_[eid].first;
                        const uint32_t v2 = edges_[eid].second;
                        if(k_max[v1] >= current_k && k_max[v2] >= current_k){
                            k_adj_in[v2].push_back({v1, eid});
                            k_adj_out[v1].push_back({v2, eid});
                            cur_k_list.push_back(edges_[eid]);
                        }
                    }
                    for(uint32_t vid = 0; vid < n_; ++vid){
                        if(k_max[vid] >= current_k){
                            is_in_subgraph[vid] = true;
                            if(l_max[vid][current_k] > sub_out_coreness[vid]){
                                sub_out_coreness[vid] = l_max[vid][current_k];
                            }
                            else{
                                sub_out_coreness[vid] = k_adj_out[vid].size();
                            }
                        }
                    }
                    //parallel h-index based method
                    bool flag = true;
                    while (flag){
                        flag = false;
                        #pragma omp parallel for num_threads(lmax_number_of_threads)
                        for(uint32_t vid = 0; vid < n_; ++vid){
                            if(is_in_subgraph[vid]){
                                vector<uint32_t> tmp_neighbor_out_coreness(k_adj_out[vid].size(),0);
                                for(uint32_t i = 0; i < k_adj_out[vid].size(); ++i){
                                    tmp_neighbor_out_coreness[i] = sub_out_coreness[k_adj_out[vid][i].vid];
                                }
                                if(hIndex(tmp_neighbor_out_coreness) < sub_out_coreness[vid]){
                                    sub_out_coreness[vid] = hIndex(tmp_neighbor_out_coreness);
                                    flag = true;
                                }
                            }
                        }
                    }
                    for(uint32_t vid = 0; vid < n_; ++vid){
                        if(is_in_subgraph[vid]){
                            l_max[vid][current_k] = sub_out_coreness[vid];
                        }
                    }

                }
            }
            if(!use_hindex){
                //using peeling method to update the l_{max} value of (M_,0)-core
                // initialize adjacency arrays for the current k-list
                for(uint32_t current_k = M_  ; current_k <= M_ ; ++current_k){
                    vector<vector<ArrayEntry>> k_adj_in, k_adj_out;
                    // 1 map the node id in k-list -> consistent, zero-based node id
                    set<::uint32_t> node_recorder;
                    map<::uint32_t,uint32_t> node_map, node_map_reverse;
                    vector<pair<uint32_t,uint32_t>> cur_k_list;
                    ::uint32_t tmp_n_ = 0;       //the number of vertices in the current k-list
                    ::uint32_t tmp_m_ = 0;          //the number of edges in the current k-list
                    for (uint32_t eid = 0; eid < edges_.size(); ++eid) {
                        const uint32_t v1 = edges_[eid].first;
                        const uint32_t v2 = edges_[eid].second;
                        if(k_max[v1] >= current_k && k_max[v2] >= current_k){
                            ++tmp_m_;
                            cur_k_list.push_back(edges_[eid]);
                            node_recorder.insert(v1);
                            node_recorder.insert(v2);
                        }
                    }
                    for (uint32_t vid : node_recorder) {
                        node_map[vid] = std::distance(node_recorder.begin(), node_recorder.find(vid));
                        node_map_reverse[node_map[vid]] = vid;    //map the consistent, zero-based node id -> node id in k-list
                    }
                    tmp_n_ = node_recorder.size();       //the number of vertices in the current k-list
                    k_adj_in.resize(tmp_n_);
                    k_adj_out.resize(tmp_n_);
                    // 2 initialize the adjacency arrays
                    for (uint32_t eid = 0; eid < cur_k_list.size(); ++eid) {
                        const uint32_t v1 = cur_k_list[eid].first;
                        const uint32_t v2 = cur_k_list[eid].second;
                        k_adj_out[node_map[v1]].push_back({node_map[v2], eid });
                        k_adj_in[node_map[v2]].push_back({node_map[v1], eid });
                    }
                    for (uint32_t vid = 0; vid < tmp_n_; ++vid) {
                        std::sort(k_adj_out[vid].begin(), k_adj_out[vid].end(),
                                  [](const ArrayEntry& ae1, const ArrayEntry& ae2) {
                                      return ae1.vid < ae2.vid;
                                  });
                        std::sort(k_adj_in[vid].begin(), k_adj_in[vid].end(),
                                  [](const ArrayEntry& ae1, const ArrayEntry& ae2) {
                                      return ae1.vid < ae2.vid;
                                  });
                    }
                    // bin-sort peeling for k-list w.r.t the out-degree, get [(k,l)-core] for all the l values
                    // note that when we delete v, for its in-neighbor w, we maintain their sub_out_coreness(a_{w}^{k})
                    // for its out-neighbor u, we maintain their in_degree in current k-list, if the in_degree < k after v's deletion
                    // then we have a_{v}^{k} = a_{u}^{k}, and we delete u as well.

                    // 1.initialization
                    std::vector<::uint32_t> deg_in(tmp_n_, 0), deg_out(tmp_n_, 0);
                    std::vector<::uint32_t> sub_out_coreness(tmp_n_, 0);
                    std::vector<bool> deleted(tmp_m_, false);  //the edge is deleted or not?
                    // 2. get the in-degree
                    ::uint32_t max_out_deg = 0;
                    for (uint32_t vid = 0; vid < tmp_n_; ++vid) {
                        deg_out[vid] = k_adj_out[vid].size();
                        max_out_deg = std::max(max_out_deg, deg_out[vid]);
                        deg_in[vid] = k_adj_in[vid].size();
                    }

                    // 3. initialize buckets
                    vector<unordered_set<::uint32_t>> buckets(max_out_deg + 1);
                    for (uint32_t vid = 0; vid < tmp_n_; ++vid) {
                        ASSERT_MSG(deg_out[vid] <= max_out_deg, "wrong sub-out-degree error " << current_k << " vid " << vid << " " << deg_out[vid] << " " << max_out_deg);
                        buckets[deg_out[vid]].insert(vid);
                    }

                    // 4. peel vertices with out-degree less than k
                    for (::uint32_t k = 0; k < buckets.size(); ++k) {
                        if(buckets[k].empty()) continue;
                        while (!buckets[k].empty()) {
                            uint32_t vid = *buckets[k].begin();
                            buckets[k].erase(buckets[k].begin());
                            sub_out_coreness[vid] = k;
                            for (const ArrayEntry &ae: k_adj_in[vid]) {   //for in neighbors, decrease their out-degree
                                ::uint32_t nbr = ae.vid;
                                if(!(deleted[ae.eid])){
                                    deleted[ae.eid] = true;
                                    if(deg_out[nbr] > k && deg_in[nbr] >= current_k) {
                                        buckets[deg_out[nbr]].erase(nbr);
                                        --deg_out[nbr];
                                        buckets[deg_out[nbr]].insert(nbr);
                                    }
                                }
                            }
                            for (const ArrayEntry &ae: k_adj_out[vid]) {  //for out neighbors, check their in-degree to see if it >= k
                                ::uint32_t nbr = ae.vid;
                                if (!(deleted[ae.eid])) {
                                    deleted[ae.eid] = true;
                                    --deg_in[nbr];
                                    if(deg_in[nbr] < current_k){
                                        ASSERT_MSG(deg_out[nbr] >= k, "wrong out degree error " << k << " vid " << nbr << " " << deg_out[nbr] << " " << k);
                                        buckets[deg_out[nbr]].erase(nbr);
                                        buckets[k].insert(nbr);
                                    }
                                }
                            }
                        }
                    }
                    //record the (current_k, a_{u}^{current_k}) pair for all vertices
                    for (uint32_t vid = 0; vid < tmp_n_; ++vid) {
                        uint32_t u = node_map_reverse[vid];
                        l_max[u][current_k] = sub_out_coreness[vid];
                    }
                }
            }


        }
        else{
            vector<vector<vector<ArrayEntry>>> total_adj_in(M_ ), total_adj_out(M_ ); //all the adj list of (k,0)-cores with 0 <= k <= M
            #pragma omp parallel for num_threads(lmax_number_of_threads)
            for(int k = 0; k < M_; ++k){
                total_adj_in[k].resize(n_);
                total_adj_out[k].resize(n_);
                for (uint32_t eid = 0; eid < edges_.size(); ++eid) {
                    const uint32_t v1 = edges_[eid].first;
                    const uint32_t v2 = edges_[eid].second;
                    if(k_max[v1] >= k && k_max[v2] >= k){
                        total_adj_in[k][v2].push_back({v1, eid});
                        total_adj_out[k][v1].push_back({v2, eid});
                    }
                }
            }


            //#pragma omp parallel for num_threads(lmax_number_of_threads)
            for(uint32_t k = 0; k < M_; ++k) {
                /*construct adj list in the (k,0)-core*/
                vector<vector<ArrayEntry>> k_adj_in(n_), k_adj_out(n_);
                k_adj_in = total_adj_in[k], k_adj_out = total_adj_out[k];
                // for (uint32_t eid = 0; eid < edges_.size(); ++eid) {
                //     const uint32_t v1 = edges_[eid].first;
                //     const uint32_t v2 = edges_[eid].second;
                //     if(k_max[v1] >= k && k_max[v2] >= k){
                //         k_adj_in[v2].push_back({v1, eid});
                //         k_adj_out[v1].push_back({v2, eid});
                //     }
                // }

                /*use the DFS-based method to maintain l_{max}(v,k) value of vertices*/
                uint32_t root = modified_edges[0].first;
                if (l_max[modified_edges[0].second][k] < l_max[modified_edges[0].first][k]){
                    root = modified_edges[0].second;
                }
                uint32_t k_M_ = l_max[root][k];

                /*find in-core of deleted edges*/
                vector<uint32_t> m_d(n_, 0);    //candidate degree of vertices
                vector<vector<ArrayEntry>> sub_adj_in(n_), sub_adj_out(n_);  //record structure of out-core graph
                vector<bool> be_in_outcore(n_, false);
                if(l_max[modified_edges[0].second][k] == l_max[modified_edges[0].first][k]){
                    //ASSERT_MSG(true, "deleting edge resulting disconnected graph");
                    lMaxFindOutcore(modified_edges[0].first,  m_d, sub_adj_in, sub_adj_out, be_in_outcore,
                                    k, k_M_, k_adj_out);
                    lMaxFindOutcore(modified_edges[0].second , m_d, sub_adj_in, sub_adj_out, be_in_outcore,
                                    k, k_M_, k_adj_out);
                }
                else{
                    lMaxFindOutcore(root ,m_d, sub_adj_in, sub_adj_out, be_in_outcore,
                                    k, k_M_, k_adj_out);
                }

                /*evict disqualified vertices using bin-sort, similar to decomposition algo*/
                vector<unordered_set<uint32_t>> buckets(*max_element(m_d.begin(), m_d.end()) + 1);
                vector<bool> deleted(m_, false);
                for(uint32_t vid = 0; vid < n_; ++vid){
                    if(be_in_outcore[vid]){
                        ASSERT_MSG(m_d[vid] >= 0, "m_d[vid] is out of range: " << m_d[vid]);
                        buckets[m_d[vid]].insert(vid);
                    }
                }

                for (::uint32_t b = 0; b < buckets.size(); ++b) {
                    if(buckets[b].empty()){
                        continue;
                    }
                    if(b >= k_M_){
                        break;
                    }
                    for (auto vid: buckets[b]) {
                        if(be_in_outcore[vid]){
                            if(m_d[vid] < k_M_){
                                l_max[vid][k]--;
                                //dif_kmax_M_group.push_back(vid);
                                for (auto in_neighbor: sub_adj_in[vid]) {
                                    if(!deleted[in_neighbor.eid]){
                                        deleted[in_neighbor.eid] = true;
                                    }
                                    if(m_d[in_neighbor.vid] > m_d[vid]){
                                        buckets[m_d[in_neighbor.vid]].erase(in_neighbor.vid);
                                        m_d[in_neighbor.vid]--;
                                        buckets[m_d[in_neighbor.vid]].insert(in_neighbor.vid);
                                    }
                                }
                                for(auto out_neighbor : sub_adj_out[vid]){
                                    if(!deleted[out_neighbor.eid]){
                                        deleted[out_neighbor.eid] = true;
                                    }
                                }
                            } else{
                                break;
                            }
                        }
                    }
                }
            }

            const auto test1 = std::chrono::steady_clock::now();
            if(use_hindex){
                //using parallel-h-index based method to update the l_{max} value of (M_,0)-core
                for(uint32_t current_k = M_  ; current_k <= M_ ; ++current_k){
                    vector<vector<ArrayEntry>> k_adj_in(n_), k_adj_out(n_);
                    vector<pair<uint32_t,uint32_t>> cur_k_list;
                    vector<bool> is_in_subgraph (n_, false);   //wether the vertex is in the current (M_,0)-core
                    vector<uint32_t> sub_out_coreness(n_, 0);  //the l_{max} value of vertices in the current (M_,0)-core
                    for (uint32_t eid = 0; eid < edges_.size(); ++eid) {
                        const uint32_t v1 = edges_[eid].first;
                        const uint32_t v2 = edges_[eid].second;
                        if(k_max[v1] >= current_k && k_max[v2] >= current_k){
                            k_adj_in[v2].push_back({v1, eid});
                            k_adj_out[v1].push_back({v2, eid});
                            cur_k_list.push_back(edges_[eid]);
                        }
                    }
                    for(uint32_t vid = 0; vid < n_; ++vid){
                        if(k_max[vid] >= current_k){
                            is_in_subgraph[vid] = true;
                            if(l_max[vid][current_k] > sub_out_coreness[vid]){
                                sub_out_coreness[vid] = l_max[vid][current_k];
                            }
                            else{
                                sub_out_coreness[vid] = k_adj_out[vid].size();
                            }
                        }
                    }
                    //parallel h-index based method
                    bool flag = true;
                    while (flag){
                        flag = false;
                        #pragma omp parallel for num_threads(lmax_number_of_threads)
                        for(uint32_t vid = 0; vid < n_; ++vid){
                            if(is_in_subgraph[vid]){
                                vector<uint32_t> tmp_neighbor_out_coreness(k_adj_out[vid].size(),0);
                                for(uint32_t i = 0; i < k_adj_out[vid].size(); ++i){
                                    tmp_neighbor_out_coreness[i] = sub_out_coreness[k_adj_out[vid][i].vid];
                                }
                                if(hIndex(tmp_neighbor_out_coreness) < sub_out_coreness[vid]){
                                    sub_out_coreness[vid] = hIndex(tmp_neighbor_out_coreness);
                                    flag = true;
                                }
                            }
                        }
                    }
                    for(uint32_t vid = 0; vid < n_; ++vid){
                        if(is_in_subgraph[vid]){
                            l_max[vid][current_k] = sub_out_coreness[vid];
                        }
                    }

                }
            }
            if(!use_hindex){
                //using peeling method to update the l_{max} value of (M_,0)-core
                // initialize adjacency arrays for the current k-list
                for(uint32_t current_k = M_  ; current_k <= M_ ; ++current_k){
                    vector<vector<ArrayEntry>> k_adj_in, k_adj_out;
                    // 1 map the node id in k-list -> consistent, zero-based node id
                    set<::uint32_t> node_recorder;
                    map<::uint32_t,uint32_t> node_map, node_map_reverse;
                    vector<pair<uint32_t,uint32_t>> cur_k_list;
                    ::uint32_t tmp_n_ = 0;       //the number of vertices in the current k-list
                    ::uint32_t tmp_m_ = 0;          //the number of edges in the current k-list
                    for (uint32_t eid = 0; eid < edges_.size(); ++eid) {
                        const uint32_t v1 = edges_[eid].first;
                        const uint32_t v2 = edges_[eid].second;
                        if(k_max[v1] >= current_k && k_max[v2] >= current_k){
                            ++tmp_m_;
                            cur_k_list.push_back(edges_[eid]);
                            node_recorder.insert(v1);
                            node_recorder.insert(v2);
                        }
                    }
                    for (uint32_t vid : node_recorder) {
                        node_map[vid] = std::distance(node_recorder.begin(), node_recorder.find(vid));
                        node_map_reverse[node_map[vid]] = vid;    //map the consistent, zero-based node id -> node id in k-list
                    }
                    tmp_n_ = node_recorder.size();       //the number of vertices in the current k-list
                    k_adj_in.resize(tmp_n_);
                    k_adj_out.resize(tmp_n_);
                    // 2 initialize the adjacency arrays
                    for (uint32_t eid = 0; eid < cur_k_list.size(); ++eid) {
                        const uint32_t v1 = cur_k_list[eid].first;
                        const uint32_t v2 = cur_k_list[eid].second;
                        k_adj_out[node_map[v1]].push_back({node_map[v2], eid });
                        k_adj_in[node_map[v2]].push_back({node_map[v1], eid });
                    }
                    for (uint32_t vid = 0; vid < tmp_n_; ++vid) {
                        std::sort(k_adj_out[vid].begin(), k_adj_out[vid].end(),
                                  [](const ArrayEntry& ae1, const ArrayEntry& ae2) {
                                      return ae1.vid < ae2.vid;
                                  });
                        std::sort(k_adj_in[vid].begin(), k_adj_in[vid].end(),
                                  [](const ArrayEntry& ae1, const ArrayEntry& ae2) {
                                      return ae1.vid < ae2.vid;
                                  });
                    }
                    // bin-sort peeling for k-list w.r.t the out-degree, get [(k,l)-core] for all the l values
                    // note that when we delete v, for its in-neighbor w, we maintain their sub_out_coreness(a_{w}^{k})
                    // for its out-neighbor u, we maintain their in_degree in current k-list, if the in_degree < k after v's deletion
                    // then we have a_{v}^{k} = a_{u}^{k}, and we delete u as well.

                    // 1.initialization
                    std::vector<::uint32_t> deg_in(tmp_n_, 0), deg_out(tmp_n_, 0);
                    std::vector<::uint32_t> sub_out_coreness(tmp_n_, 0);
                    std::vector<bool> deleted(tmp_m_, false);  //the edge is deleted or not?
                    // 2. get the in-degree
                    ::uint32_t max_out_deg = 0;
                    for (uint32_t vid = 0; vid < tmp_n_; ++vid) {
                        deg_out[vid] = k_adj_out[vid].size();
                        max_out_deg = std::max(max_out_deg, deg_out[vid]);
                        deg_in[vid] = k_adj_in[vid].size();
                    }

                    // 3. initialize buckets
                    vector<unordered_set<::uint32_t>> buckets(max_out_deg + 1);
                    for (uint32_t vid = 0; vid < tmp_n_; ++vid) {
                        ASSERT_MSG(deg_out[vid] <= max_out_deg, "wrong sub-out-degree error " << current_k << " vid " << vid << " " << deg_out[vid] << " " << max_out_deg);
                        buckets[deg_out[vid]].insert(vid);
                    }

                    // 4. peel vertices with out-degree less than k
                    for (::uint32_t k = 0; k < buckets.size(); ++k) {
                        if(buckets[k].empty()) continue;
                        while (!buckets[k].empty()) {
                            uint32_t vid = *buckets[k].begin();
                            buckets[k].erase(buckets[k].begin());
                            sub_out_coreness[vid] = k;
                            for (const ArrayEntry &ae: k_adj_in[vid]) {   //for in neighbors, decrease their out-degree
                                ::uint32_t nbr = ae.vid;
                                if(!(deleted[ae.eid])){
                                    deleted[ae.eid] = true;
                                    if(deg_out[nbr] > k && deg_in[nbr] >= current_k) {
                                        buckets[deg_out[nbr]].erase(nbr);
                                        --deg_out[nbr];
                                        buckets[deg_out[nbr]].insert(nbr);
                                    }
                                }
                            }
                            for (const ArrayEntry &ae: k_adj_out[vid]) {  //for out neighbors, check their in-degree to see if it >= k
                                ::uint32_t nbr = ae.vid;
                                if (!(deleted[ae.eid])) {
                                    deleted[ae.eid] = true;
                                    --deg_in[nbr];
                                    if(deg_in[nbr] < current_k){
                                        ASSERT_MSG(deg_out[nbr] >= k, "wrong out degree error " << k << " vid " << nbr << " " << deg_out[nbr] << " " << k);
                                        buckets[deg_out[nbr]].erase(nbr);
                                        buckets[k].insert(nbr);
                                    }
                                }
                            }
                        }
                    }
                    //record the (current_k, a_{u}^{current_k}) pair for all vertices
                    for (uint32_t vid = 0; vid < tmp_n_; ++vid) {
                        uint32_t u = node_map_reverse[vid];
                        l_max[u][current_k] = sub_out_coreness[vid];
                    }
                }
            }
            const auto test2 = std::chrono::steady_clock::now();
            //printf("DFS-based delete peeling in costs \x1b[1;31m%f\x1b[0m ms\n",
            //       std::chrono::duration<double, std::milli>(test2 - test1).count());
        }

        }

}

/**
    @brief insert a new edge into the graph and initialize the related value
    @param inserted edge
 **/
void DfsSearch::insertEdge(const pair<uint32_t, uint32_t> &edge) {
    ASSERT_MSG(std::find(edges_.begin(), edges_.end(), edge) == edges_.end(), "inserting existing edge ");

    edges_.push_back(edge);
    ++m_;
    //M_ = min(k_max[edge.first], k_max[edge.second]);
    //N_ = max(k_max[edge.first], k_max[edge.second]);

    adj_out[edge.first].push_back({edge.second, m_ - 1});
    adj_in[edge.second].push_back({edge.first, m_ - 1});

    for (uint32_t vid = 0; vid < n_; ++vid) {
        std::sort(adj_in[vid].begin(), adj_in[vid].end(),
                  [](const ArrayEntry& ae1, const ArrayEntry& ae2) {
                      return ae1.vid < ae2.vid;
                  });
        std::sort(adj_out[vid].begin(), adj_out[vid].end(),
                  [](const ArrayEntry& ae1, const ArrayEntry& ae2) {
                      return ae1.vid < ae2.vid;
                  });
    }
}


/**
    @brief delete a old edge from the graph and initialize the related value
    @param deleted edge
 **/
void DfsSearch::deleteEdge(const pair<uint32_t, uint32_t> &edge) {
    ASSERT_MSG(std::find(edges_.begin(), edges_.end(), edge) != edges_.end(), "deleting non-exsiting edge ");
    ASSERT_MSG(adj_out[edge.first].size() > 1 && adj_in[edge.second].size() > 1, "deleting edge will cause the graph disconnected");

    edges_.erase(std::remove(edges_.begin(), edges_.end(), edge), edges_.end());
    --m_;
//    M_ = min(k_max[edge.first], k_max[edge.second]);
//    N_ = max(k_max[edge.first], k_max[edge.second]);

    for(uint32_t i = 0; i < adj_out[edge.first].size(); ++i){
        if(adj_out[edge.first][i].vid == edge.second){
            adj_out[edge.first].erase(adj_out[edge.first].begin() + i);
            break;
        }
    }

    for(uint32_t i = 0; i < adj_in[edge.second].size(); ++i){
        if(adj_in[edge.second][i].vid == edge.first){
            adj_in[edge.second].erase(adj_in[edge.second].begin() + i);
            break;
        }
    }

    for (uint32_t vid = 0; vid < n_; ++vid) {
        std::sort(adj_in[vid].begin(), adj_in[vid].end(),
                  [](const ArrayEntry& ae1, const ArrayEntry& ae2) {
                      return ae1.vid < ae2.vid;
                  });
        std::sort(adj_out[vid].begin(), adj_out[vid].end(),
                  [](const ArrayEntry& ae1, const ArrayEntry& ae2) {
                      return ae1.vid < ae2.vid;
                  });
    }

}

 /**
    @brief return the D-core decomposition result based on the updated kmax/lmax value
    @param
**/
vector<vector<pair<uint32_t, uint32_t>>> DfsSearch::getDcoreDecomposition(){
    vector<vector<pair<uint32_t, uint32_t>>> d_core_decomposition(n_);
    for(uint32_t vid = 0; vid < n_; ++vid){
        for(uint32_t k = 0; k <= k_max[vid]; ++k){
            d_core_decomposition[vid].emplace_back(k, l_max[vid][k]);
        }
        reverse(d_core_decomposition[vid].begin(), d_core_decomposition[vid].end());
    }
    return d_core_decomposition;
}

