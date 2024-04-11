
#include "hIndex.h"

hIndex::hIndex(vector<pair<uint32_t, uint32_t>> &vEdges, const vector<vector<pair<::uint32_t, ::uint32_t>>> &old_d_core_decomposition) {

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
    @brief return h-index of a vector
    @param vector: a vector of uint32_t
**/
uint32_t hIndex::cal_hIndex(const vector<uint32_t> &input_vector){
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
    @brief insert a new edge into the graph and initialize the related value
    @param inserted edge
 **/
void hIndex::insertEdge(const pair<uint32_t, uint32_t> &edge) {
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
void hIndex::deleteEdge(const pair<uint32_t, uint32_t> &edge) {
    ASSERT_MSG(std::find(edges_.begin(), edges_.end(), edge) != edges_.end(), "deleting non-exsiting edge ");
    //ASSERT_MSG(adj_out[edge.first].size() > 1 && adj_in[edge.second].size() > 1, "deleting edge will cause the graph disconnected");

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
    @brief when maintain the kmax value of vertices with edge insertion, remove disqualified vertices
    @param candidate degree of vertices, bool: whether the vertex is deleted, uint32_t: the smaller k_max value of inserted/deleted edge before graph modification, uint32_t: current vertex id
           M/N: the smaller/bigger k_max value of modified edge before graph modification
**/

void hIndex::kMaxRemove (vector<uint32_t> &m_d, vector<bool> &vbDeleted, uint32_t cur_node, const uint32_t &M_){
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
void hIndex::kMaxFindIncore(uint32_t root_node_id, vector<uint32_t> &m_d,
                               vector<vector<hIndex::ArrayEntry>> &sub_adj_in,
                               vector<vector<hIndex::ArrayEntry>> &sub_adj_out, vector<bool> &be_in_incore,
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
                    //sub_adj_in[cur_node].push_back({in_neighbor.vid, in_neighbor.eid});
                    //sub_adj_out[in_neighbor.vid].push_back({cur_node, in_neighbor.eid});
                }
            }
        }
    }

}


/**
    @brief maintain the kmax value of vertices using H-index method, is_insert = true means insertion, otherwise deletion
    @param inserted/deleted edges, bool: is_insertion, M/N: the smaller/bigger k_max value of the endpoints of modified edge before graph modification
**/
void hIndex::maintainKmax(const vector<pair<uint32_t, uint32_t>> &modified_edges, bool is_insert,
                          const int & lmax_number_of_threads) {
    //edge insertion
    //for the h-index-based algorithm, both single edge and multiple edge can be processed
    //ASSERT_MSG(modified_edges.size()==1, "the number of modified edges is not equal to 1: " << modified_edges.size());
    if (is_insert) {
        vector<bool> compute(n_, false);  //needs to be computed
        vector<bool> be_in_incore(n_, false);

         auto test1 = omp_get_wtime();
        /*calculate ED value of vertices*/
        vector<uint32_t> mED(n_, 0), mPED(n_, 0);
        #pragma omp parallel for num_threads(lmax_number_of_threads)
        for (uint32_t vid = 0; vid < n_; ++vid) {
            for (auto neighbors: adj_in[vid]){
                if (k_max[neighbors.vid] >= k_max[vid]) {
                    ++mED[vid];
                }
            }
        }
        /*calculate PED value of vertices*/
        #pragma omp parallel for num_threads(lmax_number_of_threads)
        for (uint32_t vid = 0; vid < n_; ++vid) {
            mPED[vid] = mED[vid];
            if(!(mED[vid] == 0)){
                for (auto neighbors: adj_in[vid]){
                    if(k_max[neighbors.vid] == k_max[vid] && mED[neighbors.vid] <= k_max[vid]){
                        --mPED[vid];
                    }
                }
            }
        }
        auto test2 = omp_get_wtime();
        
        vector<uint32_t> original_kmax = k_max;

        /*find in-core of root*/
        #pragma omp parallel for num_threads(lmax_number_of_threads)
        for (auto &edge: modified_edges) {
            //vector<uint32_t> dif_kmax_M_group; // the set of vertices have their kmax changed after Kmax value maintenance
            uint32_t root = edge.first;
            if (k_max[edge.second] < k_max[edge.first]) {
                root = edge.second;
            }
            //#pragma omp parallel for num_threads(lmax_number_of_threads)
            for(uint32_t vid = 0; vid < n_; ++vid){
                if(k_max[vid] == k_max[root] && mPED[vid] > k_max[vid]){
                    compute[vid] = true;
                    //k_max[vid] = adj_in[vid].size();
                    ++k_max[vid];
                }
            }
        }
         auto test3 = omp_get_wtime();

        /*do the initialization*/
        bool flag = true;
        uint32_t round_cnt = 0;
        while (flag){
            flag = false;
            #pragma omp parallel for num_threads(lmax_number_of_threads)
            for(uint32_t vid = 0; vid < n_; ++vid){
                if(compute[vid]){
                    vector<uint32_t> tmp_neighbor_in_coreness(adj_in[vid].size(),0);
                    for(uint32_t i = 0; i < adj_in[vid].size(); ++i){
                        if(k_max[adj_in[vid][i].vid] >= original_kmax[vid]){
                            tmp_neighbor_in_coreness[i] = k_max[adj_in[vid][i].vid];
                        }
                    }
                    uint32_t tmp_h_index = cal_hIndex(tmp_neighbor_in_coreness);
                    if(tmp_h_index < k_max[vid]){
                        k_max[vid] = tmp_h_index;
                        flag = true;
                    }
                }
            }
            round_cnt++;
        }
         auto test4 = omp_get_wtime();

        // modify l_max value based on k_max value
        #pragma omp parallel for num_threads(lmax_number_of_threads)
        for(uint32_t vid = 0; vid < n_; ++vid){
            if(k_max[vid] > original_kmax[vid]){
                max_k_max = max(max_k_max, k_max[vid]);
                for(uint32_t i = 0; i < k_max[vid] - original_kmax[vid];i++){
                    l_max[vid].push_back(0);
                }
            }
            // ASSERT_MSG(k_max[vid] == l_max[vid].size() - 1, 
            //             "insert k_max and l_max inconsistent: " << original_kmax[vid] << " " <<  k_max[vid] << " " << l_max[vid].size());
        }
         auto test5 = omp_get_wtime();

        // printf("Insertion kmax test1 \x1b[1;31m%f\x1b[0m ms; test2 \x1b[1;31m%f\x1b[0m ms; test3 \x1b[1;31m%f\x1b[0m ms;"
        //        "test4 \x1b[1;31m%f\x1b[0m ms\n",
        //         (test2 - test1)*1000,
        //         (test3 - test2)*1000,
        //         (test4 - test3)*1000,
        //        (test5 - test4)*1000);

    }
        //edge deletion
    else{
        vector<bool> compute(n_, false);  //needs to be computed
        vector<bool> be_in_incore(n_, false);
        auto test1 = omp_get_wtime();
        /*calculate ED value of vertices*/
        vector<uint32_t> mED(n_, 0);
        #pragma omp parallel for num_threads(lmax_number_of_threads)
        for (uint32_t vid = 0; vid < n_; ++vid) {
            for (auto neighbors: adj_in[vid]) {
                if (k_max[neighbors.vid] >= k_max[vid]) {
                    ++mED[vid];
                }
            }
        }
        
        vector<uint32_t> original_kmax = k_max;
        auto test2 = omp_get_wtime();

        /*find in-core of root*/
        #pragma omp parallel for num_threads(lmax_number_of_threads)
        for (const auto &edge: modified_edges) {
            uint32_t root = edge.first;
            if (k_max[edge.second] < k_max[edge.first]) {
                root = edge.second;
            }

            //#pragma omp parallel for num_threads(lmax_number_of_threads)
            for(uint32_t vid = 0; vid < n_; ++vid){
                if(k_max[vid] == k_max[root] && mED[vid] <= k_max[vid]){
                    compute[vid] = true;
                    //k_max[vid] = adj_in[vid].size();
                   if(k_max[vid] > 0){
                       --k_max[vid];
                   }
                   else{
                       k_max[vid] = 0;
                   }

                }
            }
        }
        auto test3 = omp_get_wtime();
        /*do the initialization*/
       

        bool flag = true;
        uint32_t round_cnt = 0;
        while (flag){
            flag = false;
            #pragma omp parallel for num_threads(lmax_number_of_threads)
            for(uint32_t vid = 0; vid < n_; ++vid){
                if(compute[vid]){
                    vector<uint32_t> tmp_neighbor_in_coreness(adj_in[vid].size(),0);
                    for(uint32_t i = 0; i < adj_in[vid].size(); ++i){
                        if(k_max[adj_in[vid][i].vid] >= original_kmax[vid]){
                            tmp_neighbor_in_coreness[i] = k_max[adj_in[vid][i].vid];
                        }
                        //tmp_neighbor_in_coreness[i] = k_max[adj_in[vid][i].vid];
                    }
                    uint32_t tmp_h_index = cal_hIndex(tmp_neighbor_in_coreness);
                    if(tmp_h_index < k_max[vid]){
                        k_max[vid] = tmp_h_index;
                        flag = true;
                    }
                }
            }
            round_cnt++;
        }

        auto test4 = omp_get_wtime();
        // modify l_max value based on k_max value
        #pragma omp parallel for num_threads(lmax_number_of_threads)
        for(uint32_t vid = 0; vid < n_; ++vid){
            if(k_max[vid] < original_kmax[vid]){
                if(max_k_max == k_max[vid] && max_k_max > 0){
                    --max_k_max;
                }
                for(uint32_t i = 0; i < original_kmax[vid] - k_max[vid] ;++i){
                    l_max[vid].pop_back();
                }
                
            }
            // ASSERT_MSG(k_max[vid] == l_max[vid].size() - 1, 
            //             "delete k_max and l_max inconsistent: " << original_kmax[vid] << " " << k_max[vid] << " " << l_max[vid].size());
        }
        auto test5 = omp_get_wtime();
        // printf("Deletion kmax test1 \x1b[1;31m%f\x1b[0m ms; test2 \x1b[1;31m%f\x1b[0m ms; test3 \x1b[1;31m%f\x1b[0m ms;"
        //        "test4 \x1b[1;31m%f\x1b[0m ms\n",
        //         (test2 - test1)*1000,
        //         (test3 - test2)*1000,
        //         (test4 - test3)*1000,
        //        (test5 - test4)*1000);
    }
}


/**
    @brief maintain the kmax value of vertices using DFS search, is_insert = true means insertion, otherwise deletion
    @param inserted/deleted edges, bool: is_insertion, M/N: the smaller/bigger k_max value of the endpoints of modified edge before graph modification
**/

void hIndex::maintainKmaxDfs( const vector<pair<uint32_t, uint32_t>> &modified_edges, bool is_insert, const uint32_t &M_) {
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
    @brief maintain the kmax value of vertices using H-index method, is_insert = true means insertion, otherwise deletion
    @param inserted/deleted edges, bool: is_insertion, M/N: the smaller/bigger k_max value of the endpoints of modified edge before graph modification
**/
void hIndex::maintainKmaxSingle(const vector<pair<uint32_t, uint32_t>> &modified_edges, bool is_insert,
                          const int & lmax_number_of_threads) {
    //edge insertion
    //for the h-index-based algorithm, both single edge and multiple edge can be processed
    //ASSERT_MSG(modified_edges.size()==1, "the number of modified edges is not equal to 1: " << modified_edges.size());
    if (is_insert) {
        vector<bool> compute(n_, false);  //needs to be computed
        vector<bool> be_in_incore(n_, false);

        auto test1 = omp_get_wtime();
        /*calculate ED value of vertices*/
        vector<uint32_t> mED(n_, 0), mPED(n_, 0);
        #pragma omp parallel for num_threads(lmax_number_of_threads)
        for (uint32_t vid = 0; vid < n_; ++vid) {
            for (auto neighbors: adj_in[vid]) {
                if (k_max[neighbors.vid] >= k_max[vid]) {
                    ++mED[vid];
                }
            }
        }
        /*calculate PED value of vertices*/
        #pragma omp parallel for num_threads(lmax_number_of_threads)
        for (uint32_t vid = 0; vid < n_; ++vid) {
            mPED[vid] = mED[vid];
            for (auto neighbors: adj_in[vid]) {
                if(k_max[neighbors.vid] == k_max[vid] && mED[neighbors.vid] <= k_max[vid]){
                    --mPED[vid];
                }
            }
        }
        auto test2 = omp_get_wtime();

        vector<uint32_t> original_kmax = k_max;

        /*find in-core of root*/
        #pragma omp parallel for num_threads(lmax_number_of_threads)
        for (auto &edge: modified_edges) {
            //vector<uint32_t> dif_kmax_M_group; // the set of vertices have their kmax changed after Kmax value maintenance
            uint32_t root = edge.first;
            if (k_max[edge.second] < k_max[edge.first]) {
                root = edge.second;
            }
            //#pragma omp parallel for num_threads(lmax_number_of_threads)
            for(uint32_t vid = 0; vid < n_; ++vid){
                if(k_max[vid] == k_max[root] && mPED[vid] > k_max[vid]){
                    compute[vid] = true;
                    k_max[vid] = adj_in[vid].size();
                }
            }
        }
        auto test3 = omp_get_wtime();

        /*do the initialization*/
        bool flag = true;
        uint32_t round_cnt = 0;
        while (flag){
            flag = false;
            #pragma omp parallel for num_threads(lmax_number_of_threads)
            for(uint32_t vid = 0; vid < n_; ++vid){
                if(compute[vid]){
                    vector<uint32_t> tmp_neighbor_in_coreness(adj_in[vid].size(),0);
                    for(uint32_t i = 0; i < adj_in[vid].size(); ++i){
                        if(k_max[adj_in[vid][i].vid] >= original_kmax[vid]){
                            tmp_neighbor_in_coreness[i] = k_max[adj_in[vid][i].vid];
                        }
                    }
                    uint32_t tmp_h_index = cal_hIndex(tmp_neighbor_in_coreness);
                    if(tmp_h_index < k_max[vid]){
                        k_max[vid] = tmp_h_index;
                        flag = true;
                    }
                }
            }
            round_cnt++;
        }
        auto test4 = omp_get_wtime();

        // modify l_max value based on k_max value
        #pragma omp parallel for num_threads(lmax_number_of_threads)
        for(uint32_t vid = 0; vid < n_; ++vid){
            if(k_max[vid] > original_kmax[vid]){
                max_k_max = max(max_k_max, k_max[vid]);
                for(uint32_t i = 0; i < k_max[vid] - original_kmax[vid];i++){
                    l_max[vid].push_back(0);
                }
            }
            ASSERT_MSG(k_max[vid] == l_max[vid].size() - 1,
                       "insert k_max and l_max inconsistent: " << original_kmax[vid] << " " <<  k_max[vid] << " " << l_max[vid].size());
        }
        auto test5 = omp_get_wtime();

        // printf("Insertion test1 \x1b[1;31m%f\x1b[0m ms; test2 \x1b[1;31m%f\x1b[0m ms; test3 \x1b[1;31m%f\x1b[0m ms;"
        //        "test4\x1b[1;31m%f\x1b[0m ms\n",
        //         (test2 - test1)*1000,
        //         (test3 - test2)*1000,
        //         (test4 - test3)*1000,
        //        (test5 - test4)*1000);

    }
        //edge deletion
    else{
        vector<bool> compute(n_, false);  //needs to be computed
        vector<bool> be_in_incore(n_, false);
        auto test1 = omp_get_wtime();
        /*calculate ED value of vertices*/
        vector<uint32_t> mED(n_, 0), mPED(n_, 0);
#pragma omp parallel for num_threads(lmax_number_of_threads)
        for (uint32_t vid = 0; vid < n_; ++vid) {
            for (auto neighbors: adj_in[vid]) {
                if (k_max[neighbors.vid] >= k_max[vid]) {
                    mED[vid]++;
                }
            }
        }
        auto test2 = omp_get_wtime();
        vector<uint32_t> original_kmax = k_max;


        /*find in-core of root*/
#pragma omp parallel for num_threads(lmax_number_of_threads)
        for (const auto &edge: modified_edges) {
            uint32_t root = edge.first;
            if (k_max[edge.second] < k_max[edge.first]) {
                root = edge.second;
            }

            //#pragma omp parallel for num_threads(lmax_number_of_threads)
            for(uint32_t vid = 0; vid < n_; ++vid){
                if(k_max[vid] == k_max[root] && mED[vid] <= k_max[vid]){
                    compute[vid] = true;
                    //k_max[vid] = adj_in[vid].size();
                    if(k_max[vid] > 0){
                        --k_max[vid];
                    }
                    else{
                        k_max[vid] = 0;
                    }

                }
            }
        }
        auto test3 = omp_get_wtime();
        /*do the initialization*/


        bool flag = true;
        uint32_t round_cnt = 0;
        while (flag){
            flag = false;
#pragma omp parallel for num_threads(lmax_number_of_threads)
            for(uint32_t vid = 0; vid < n_; ++vid){
                if(compute[vid]){
                    vector<uint32_t> tmp_neighbor_in_coreness(adj_in[vid].size(),0);
                    for(uint32_t i = 0; i < adj_in[vid].size(); ++i){
                        tmp_neighbor_in_coreness[i] = k_max[adj_in[vid][i].vid];
                    }
                    uint32_t tmp_h_index = cal_hIndex(tmp_neighbor_in_coreness);
                    if(tmp_h_index < k_max[vid]){
                        k_max[vid] = tmp_h_index;
                        flag = true;
                    }
                }
            }
            round_cnt++;
        }

        auto test4 = omp_get_wtime();
        // modify l_max value based on k_max value
#pragma omp parallel for num_threads(lmax_number_of_threads)
        for(uint32_t vid = 0; vid < n_; ++vid){
            if(k_max[vid] < original_kmax[vid]){
                if(max_k_max == k_max[vid] && max_k_max > 0){
                    --max_k_max;
                }
                for(uint32_t i = 0; i < original_kmax[vid] - k_max[vid] ;++i){
                    l_max[vid].pop_back();
                }

            }
            ASSERT_MSG(k_max[vid] == l_max[vid].size() - 1,
                       "delete k_max and l_max inconsistent: " << original_kmax[vid] << " " << k_max[vid] << " " << l_max[vid].size());
        }
        auto test5 = omp_get_wtime();
        // printf("Deletion test1 \x1b[1;31m%f\x1b[0m ms; test2 \x1b[1;31m%f\x1b[0m ms; test3 \x1b[1;31m%f\x1b[0m ms;"
        //        "test4\x1b[1;31m%f\x1b[0m ms\n",
        //         (test2 - test1)*1000,
        //         (test3 - test2)*1000,
        //         (test4 - test3)*1000,
        //        (test5 - test4)*1000);
    }
}


/**
    @brief when maintain the lmax value of vertices with edge deletion, using BFS to find out-core group of the endpoints of the deleted edges
    @param m_d:candidate degree of vertices, be_in_outcore: whether the vertex is belongs to the out-core group,
           sub_adj_in/out: the adj list of the found out_core subgraph, uint32_t root_node_id: current vertex id
**/
void hIndex::lMaxFindOutcore(uint32_t root_node_id, vector<uint32_t> &m_d,
                                vector<vector<hIndex::ArrayEntry>> &sub_adj_in,
                                vector<vector<hIndex::ArrayEntry>> &sub_adj_out, vector<bool> &be_in_outcore,
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
                    //sub_adj_out[cur_node].push_back({out_neighbor.vid, out_neighbor.eid});
                    //sub_adj_in[out_neighbor.vid].push_back({cur_node, out_neighbor.eid});
                }
            }
        }
    }

}


/**
    @brief maintain the klists value of vertices based on the updated kmax value, is_insert = true means insertion, otherwise deletion,
    @param inserted/deleted edges, bool: is_insertion, M/N: the smaller/bigger k_max value of the endpoints of modified edge before graph modification
**/

void hIndex::maintainKlist(const vector<pair<uint32_t, uint32_t>> &modified_edges, bool is_insert, const uint32_t &M_, bool k0core_pruning, bool reuse_pruning,
                           const int &lmax_number_of_threads) {
    //edge insertion
    //for the basic DFS algorithm, we only consider single edge modification
    //ASSERT_MSG(modified_edges.size()==1, "the number of modified edges is not equal to 1: " << modified_edges.size());
    //if(!dif_kmax_M_group.empty()){
    if(is_insert){
        //for all the (k,0)-cores with 0 <= k <= M, we maintain the l_{max}(v, k) value of vertices
        //in the DFS search based way given edge insertion.
        if(reuse_pruning && k0core_pruning){
            //using parallel-h-index based method to update the l_{max} value
            //std::chrono::duration<double> initialzation, find_outcore, h_index_computation;
            double initialzation = 0, find_outcore = 0, h_index_computation = 0;
            vector<vector<ArrayEntry>> k_adj_in_cur(n_), k_adj_out_cur(n_);
            vector<uint32_t> mED_out_cur(n_, 0), mPED_out_cur(n_, 0);

            //first get the unique (k, 0)-cores
            vector<int> unique_k0_cores;
            set<int, greater<int>> k0_value_set;
            for (uint32_t i = 0; i < n_; ++i) {
                if(k_max[i] <= M_ + 1){
                    k0_value_set.insert(k_max[i]);
                }
            }
            for(int i : k0_value_set){
                unique_k0_cores.push_back(i);
            }
            printf("k0core pruning efficiency: %f, %d, %d\n", (float)unique_k0_cores.size() / (float)(M_ + 2), unique_k0_cores.size(), M_ + 2);

            //record the k_max value of edges
            vector<vector<int>> k_max_edges_group(M_ + 2);
            for(uint32_t eid = 0; eid < edges_.size(); ++eid){
                k_max_edges_group[min(M_ + 1, min(k_max[edges_[eid].first], k_max[edges_[eid].second]))].push_back(eid);
            }

             //record the k_max value of modified edges
            vector<vector<int>> k_max_modify_edges_group(M_ + 2);
            for(uint32_t eid = 0; eid < modified_edges.size(); ++eid){
                k_max_modify_edges_group[min(M_ + 1, min(k_max[modified_edges[eid].first], k_max[modified_edges[eid].second]))].push_back(eid);
            }

            for(uint32_t k : unique_k0_cores){
                auto test1 = omp_get_wtime();
                //vector<vector<ArrayEntry>> k_adj_in(n_), k_adj_out(n_);
                //vector<uint32_t> mED_out(n_, 0), mPED_out(n_, 0);
                vector<bool> is_in_subgraph (n_, false);   //wether the vertex is in the current (k,0)-core
                vector<uint32_t> sub_out_coreness(n_, 0);  //the l_{max} value of vertices in the current (k,0)-core
                for (uint32_t eid : k_max_edges_group[k]) {
                    const uint32_t v1 = edges_[eid].first;
                    const uint32_t v2 = edges_[eid].second;
                    k_adj_in_cur[v2].push_back({v1, eid});
                    k_adj_out_cur[v1].push_back({v2, eid});
                    if(l_max[v2][k] >= l_max[v1][k]) {
                        ++mED_out_cur[v1];
                    }
                }
                /*calculate PED value of vertices*/
                #pragma omp parallel for num_threads(lmax_number_of_threads)
                for (uint32_t vid = 0; vid < n_; ++vid) {
                    if(!k_adj_out_cur.empty()){
                        mPED_out_cur[vid] = mED_out_cur[vid];
                        for (auto neighbors: k_adj_out_cur[vid]) {
                            if(l_max[neighbors.vid][k] == l_max[vid][k] && mED_out_cur[neighbors.vid] > l_max[vid][k]){
                                --mPED_out_cur[vid];
                            }
                        }
                    }
                }



                auto test2 = omp_get_wtime();

                vector<bool> compute(n_, false);  //needs to be computed
                vector<bool> be_in_outcore(n_, false);
                /*find out-core of inserted edges*/
                 #pragma omp parallel for num_threads(lmax_number_of_threads)
                for(auto &eid : k_max_modify_edges_group[k]){
                    auto edge = modified_edges[eid];
                    uint32_t root = edge.first;
                    if (l_max[edge.second][k] < l_max[edge.first][k]) {
                        root = edge.second;
                    }
                    uint32_t k_M_ = l_max[root][k];

                    for(uint32_t vid = 0; vid < n_; ++vid){
                        if(k_max[vid] >= k && l_max[vid][k] == k_M_ && mPED_out_cur[vid] > l_max[vid][k]) {
                            compute[vid] = true;
                            l_max[vid][k] = k_adj_out_cur[vid].size();
                        }
                    }
                }
                auto test3 = omp_get_wtime();


                bool flag = true;
                uint32_t round_cnt = 0;
                while (flag){
                    flag = false;
                    #pragma omp parallel for num_threads(lmax_number_of_threads)
                    for(uint32_t vid = 0; vid < n_; ++vid){
                        if(compute[vid]){
                            vector<uint32_t> tmp_neighbor_out_coreness(k_adj_out_cur[vid].size(), 0);
                            for(uint32_t i = 0; i < k_adj_out_cur[vid].size(); ++i){
                                if(l_max[k_adj_out_cur[vid][i].vid][k] >= l_max[vid][k]){
                                    tmp_neighbor_out_coreness[i] = l_max[k_adj_out_cur[vid][i].vid][k];
                                }
                            }
                            uint32_t tmp_h_index = cal_hIndex(tmp_neighbor_out_coreness);
                            if(tmp_h_index < l_max[vid][k]){
                                l_max[vid][k] = tmp_h_index;
                                flag = true;
                            }
                        }
                    }
                    round_cnt++;
                }

                fill(mED_out_cur.begin(), mED_out_cur.end(), 0);  //reset all values to 0
                fill(mPED_out_cur.begin(), mPED_out_cur.end(), 0);


                const auto test4 = omp_get_wtime();
                initialzation += test2-test1;
                find_outcore += test3-test2;
                h_index_computation += test4-test3;
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

            printf("Insertion lmax initilization \x1b[1;31m%f\x1b[0m ms; find out-core costs \x1b[1;31m%f\x1b[0m ms; h-index computation costs \x1b[1;31m%f\x1b[0m ms \n",
                   initialzation*1000,
                   find_outcore*1000,
                   h_index_computation*1000);
        }
        else if(reuse_pruning){
            //using parallel-h-index based method to update the l_{max} value
            //std::chrono::duration<double> initialzation, find_outcore, h_index_computation;
            auto out_start = omp_get_wtime();
            double initialzation = 0, find_outcore = 0, h_index_computation = 0;
            vector<vector<ArrayEntry>> k_adj_in_cur(n_), k_adj_out_cur(n_);
            vector<uint32_t> mED_out_cur(n_, 0), mPED_out_cur(n_, 0);

            //record the k_max value of all edges
            vector<vector<int>> k_max_edges_group(M_ + 2);
            for(uint32_t eid = 0; eid < edges_.size(); ++eid){
                k_max_edges_group[min(M_ + 1, min(k_max[edges_[eid].first], k_max[edges_[eid].second]))].push_back(eid);
            }

            //record the k_max value of modified edges
            vector<vector<int>> k_max_modify_edges_group(M_ + 2);
            for(uint32_t eid = 0; eid < modified_edges.size(); ++eid){
                k_max_modify_edges_group[min(M_ + 1, min(k_max[modified_edges[eid].first], k_max[modified_edges[eid].second]))].push_back(eid);
            }

            auto out_end = omp_get_wtime();

            printf("Insertion out initialzation costs \x1b[1;31m%f\x1b[0m ms; ",
                   (out_end - out_start)*1000);

            for(int k = M_ + 1 ; k >= 0; --k){
                auto test1 = omp_get_wtime();
                //vector<vector<ArrayEntry>> k_adj_in(n_), k_adj_out(n_);
                //vector<uint32_t> mED_out(n_, 0), mPED_out(n_, 0);
                vector<bool> is_in_subgraph (n_, false);   //wether the vertex is in the current (k,0)-core
                vector<uint32_t> sub_out_coreness(n_, 0);  //the l_{max} value of vertices in the current (k,0)-core
                for (uint32_t eid : k_max_edges_group[k]) {
                    const uint32_t v1 = edges_[eid].first;
                    const uint32_t v2 = edges_[eid].second;
                    k_adj_in_cur[v2].push_back({v1, eid});
                    k_adj_out_cur[v1].push_back({v2, eid});
                    if(l_max[v2][k] >= l_max[v1][k]) {
                        ++mED_out_cur[v1];
                    }
                }
                /*calculate PED value of vertices*/
                #pragma omp parallel for num_threads(lmax_number_of_threads)
                for (uint32_t vid = 0; vid < n_; ++vid) {
                    if(!k_adj_out_cur.empty()){
                        mPED_out_cur[vid] = mED_out_cur[vid];
                        if(!(mED_out_cur[vid] == 0)){
                            for (auto neighbors: k_adj_out_cur[vid]) {
                                if(l_max[neighbors.vid][k] == l_max[vid][k] && mED_out_cur[neighbors.vid] > l_max[vid][k]){
                                    --mPED_out_cur[vid];
                                }
                            }
                        }
                       
                    }
                }

                auto test2 = omp_get_wtime();
                vector<bool> compute(n_, false);  //needs to be computed
                vector<bool> be_in_outcore(n_, false);

                /*find out-core of inserted edges*/
                #pragma omp parallel for num_threads(lmax_number_of_threads)
                for(auto &eid : k_max_modify_edges_group[k]){
                    auto edge = modified_edges[eid];
                    uint32_t root = edge.first;
                    if (l_max[edge.second][k] < l_max[edge.first][k]) {
                        root = edge.second;
                    }
                    uint32_t k_M_ = l_max[root][k];

                    for(uint32_t vid = 0; vid < n_; ++vid){
                        if(k_max[vid] >= k && l_max[vid][k] == k_M_ && mPED_out_cur[vid] > l_max[vid][k]) {
                            compute[vid] = true;
                            l_max[vid][k] = k_adj_out_cur[vid].size();
                        }
                    }
                }

                auto test3 = omp_get_wtime();


                bool flag = true;
                uint32_t round_cnt = 0;
                while (flag){
                    flag = false;
                    #pragma omp parallel for num_threads(lmax_number_of_threads)
                    for(uint32_t vid = 0; vid < n_; ++vid){
                        if(compute[vid]){
                            vector<uint32_t> tmp_neighbor_out_coreness(k_adj_out_cur[vid].size(), 0);
                            for(uint32_t i = 0; i < k_adj_out_cur[vid].size(); ++i){
                                if(l_max[k_adj_out_cur[vid][i].vid][k] >= l_max[vid][k]){
                                    tmp_neighbor_out_coreness[i] = l_max[k_adj_out_cur[vid][i].vid][k];
                                }
                            }
                            uint32_t tmp_h_index = cal_hIndex(tmp_neighbor_out_coreness);
                            if(tmp_h_index < l_max[vid][k]){
                                l_max[vid][k] = tmp_h_index;
                                flag = true;
                            }
                        }
                    }
                    round_cnt++;
                }

                fill(mED_out_cur.begin(), mED_out_cur.end(), 0);  //reset all values to 0
                fill(mPED_out_cur.begin(), mPED_out_cur.end(), 0);


                const auto test4 = omp_get_wtime();
                initialzation += test2-test1;
                find_outcore += test3-test2;
                h_index_computation += test4-test3;
            }

            printf("Insertion lmax initilization \x1b[1;31m%f\x1b[0m ms; find out-core costs \x1b[1;31m%f\x1b[0m ms; h-index computation costs \x1b[1;31m%f\x1b[0m ms \n",
                   initialzation*1000,
                   find_outcore*1000,
                   h_index_computation*1000);
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
            printf("k0core pruning efficiency: %f, %d, %d\n", (float)unique_k0_cores.size() / (float)(M_ + 1), unique_k0_cores.size(), M_ + 1);


            for(uint32_t k : unique_k0_cores){
                auto test1 = omp_get_wtime();
                vector<vector<ArrayEntry>> k_adj_in(n_), k_adj_out(n_);
                vector<uint32_t> mED_out(n_, 0), mPED_out(n_, 0);
                //vector<pair<uint32_t,uint32_t>> cur_k_list;
                vector<bool> is_in_subgraph (n_, false);   //wether the vertex is in the current (k,0)-core
                vector<uint32_t> sub_out_coreness(n_, 0);  //the l_{max} value of vertices in the current (k,0)-core
                for (uint32_t eid = 0; eid < edges_.size(); ++eid) {
                    const uint32_t v1 = edges_[eid].first;
                    const uint32_t v2 = edges_[eid].second;
                    if(k_max[v1] >= k && k_max[v2] >= k){
                        k_adj_in[v2].push_back({v1, eid});
                        k_adj_out[v1].push_back({v2, eid});
                        if(l_max[v2][k] >= l_max[v1][k]){
                            ++mED_out[v1];
                        }
                    }
                }


                /*calculate PED value of vertices*/
                #pragma omp parallel for num_threads(lmax_number_of_threads)
                for (uint32_t vid = 0; vid < n_; ++vid) {
                    if(!k_adj_out.empty()){
                        mPED_out[vid] = mED_out[vid];
                        for (auto neighbors: k_adj_out[vid]) {
                            if(l_max[neighbors.vid][k] == l_max[vid][k] && mED_out[neighbors.vid] > l_max[vid][k]){
                                --mPED_out[vid];
                            }
                        }
                        // for (auto neighbors: k_adj_out[vid]) {
                        //     if (l_max[neighbors.vid][k] > l_max[vid][k] ||
                        //         (l_max[neighbors.vid][k] == l_max[vid][k] && mED_out[neighbors.vid] > l_max[vid][k])) {
                        //         ++mPED_out[vid];
                        //     }
                        // }
                    }
                }

                auto test2 = omp_get_wtime();

                vector<bool> compute(n_, false);  //needs to be computed
                vector<bool> be_in_outcore(n_, false);
                /*find out-core of inserted edges*/


                #pragma omp parallel for num_threads(lmax_number_of_threads)
                for(auto & edge : modified_edges){
                    if(k_max[edge.first] >= k && k_max[edge.second] >= k){
                        uint32_t root = edge.first;
                        if (l_max[edge.second][k] < l_max[edge.first][k]) {
                            root = edge.second;
                        }
                        uint32_t k_M_ = l_max[root][k];

                        for(uint32_t vid = 0; vid < n_; ++vid){
                            if(l_max[vid][k] == k_M_ && mPED_out[vid] > l_max[vid][k]){
                                compute[vid] = true;
                                l_max[vid][k] = k_adj_out[vid].size();
                            }
                        }
                    }
                }
                auto test3 = omp_get_wtime();


                bool flag = true;
                uint32_t round_cnt = 0;
                while (flag){
                    flag = false;
                    #pragma omp parallel for num_threads(lmax_number_of_threads)
                    for(uint32_t vid = 0; vid < n_; ++vid){
                        if(compute[vid]){
                            vector<uint32_t> tmp_neighbor_out_coreness(k_adj_out[vid].size(),0);
                            for(uint32_t i = 0; i < k_adj_out[vid].size(); ++i){
                                if(l_max[k_adj_out[vid][i].vid][k] >= l_max[vid][k]){
                                    tmp_neighbor_out_coreness[i] = l_max[k_adj_out[vid][i].vid][k];
                                }
                            }
                            uint32_t tmp_h_index = cal_hIndex(tmp_neighbor_out_coreness);
                            if(tmp_h_index < l_max[vid][k]){
                                l_max[vid][k] = tmp_h_index;
                                flag = true;
                            }
                        }
                    }
                    round_cnt++;
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
        else{
            //using parallel-h-index based method to update the l_{max} value
            //std::chrono::duration<double> initialzation, find_outcore, h_index_computation;
            double initialzation = 0, find_outcore = 0, h_index_computation = 0;
            for(uint32_t k = 0 ; k <= M_ + 1; ++k){
                auto test1 = omp_get_wtime();
                vector<vector<ArrayEntry>> k_adj_in(n_), k_adj_out(n_);
                vector<uint32_t> mED_out(n_, 0), mPED_out(n_, 0);
                vector<bool> is_in_subgraph (n_, false);   //wether the vertex is in the current (k,0)-core
                vector<uint32_t> sub_out_coreness(n_, 0);  //the l_{max} value of vertices in the current (k,0)-core
                for (uint32_t eid = 0; eid < edges_.size(); ++eid) {
                    const uint32_t v1 = edges_[eid].first;
                    const uint32_t v2 = edges_[eid].second;
                    if(k_max[v1] >= k && k_max[v2] >= k){
                        k_adj_in[v2].push_back({v1, eid});
                        k_adj_out[v1].push_back({v2, eid});
                        if(l_max[v2][k] >= l_max[v1][k]){
                            ++mED_out[v1];
                        }
                    }
                }

                /*calculate PED value of vertices*/
                #pragma omp parallel for num_threads(lmax_number_of_threads)
                for (uint32_t vid = 0; vid < n_; ++vid) {
                    if(!k_adj_out.empty()){
                        mPED_out[vid] = mED_out[vid];
                        for (auto neighbors: k_adj_out[vid]) {
                            if(l_max[neighbors.vid][k] == l_max[vid][k] && mED_out[neighbors.vid] > l_max[vid][k]){
                                --mPED_out[vid];
                            }
                        }
                    }
                }


                auto test2 = omp_get_wtime();

                vector<bool> compute(n_, false);  //needs to be computed
                vector<bool> be_in_outcore(n_, false);
                /*find out-core of inserted edges*/


                #pragma omp parallel for num_threads(lmax_number_of_threads)
                for(auto & edge : modified_edges){
                    if(k_max[edge.first] >= k && k_max[edge.second] >= k){
                        uint32_t root = edge.first;
                        if (l_max[edge.second][k] < l_max[edge.first][k]) {
                            root = edge.second;
                        }
                        uint32_t k_M_ = l_max[root][k];

                        for(uint32_t vid = 0; vid < n_; ++vid){
                            if(k_max[vid] >= k && l_max[vid][k] == k_M_ && mPED_out[vid] > l_max[vid][k]){
                                compute[vid] = true;
                                l_max[vid][k] = k_adj_out[vid].size();
                            }
                        }
                    }
                }
                auto test3 = omp_get_wtime();


                bool flag = true;
                uint32_t round_cnt = 0;
                while (flag){
                    flag = false;
                    #pragma omp parallel for num_threads(lmax_number_of_threads)
                    for(uint32_t vid = 0; vid < n_; ++vid){
                        if(compute[vid]){
                            vector<uint32_t> tmp_neighbor_out_coreness(k_adj_out[vid].size(), 0);
                            for(uint32_t i = 0; i < k_adj_out[vid].size(); ++i){
                                if(l_max[k_adj_out[vid][i].vid][k] >= l_max[vid][k]){
                                    tmp_neighbor_out_coreness[i] = l_max[k_adj_out[vid][i].vid][k];
                                }
                            }
                            uint32_t tmp_h_index = cal_hIndex(tmp_neighbor_out_coreness);
                            if(tmp_h_index < l_max[vid][k]){
                                l_max[vid][k] = tmp_h_index;
                                flag = true;
                            }
                        }
                    }
                    round_cnt++;
                }

                const auto test4 = omp_get_wtime();
                initialzation += test2-test1;
                find_outcore += test3-test2;
                h_index_computation += test4-test3;
            }

            printf("Insertion lmax initilization \x1b[1;31m%f\x1b[0m ms; find out-core costs \x1b[1;31m%f\x1b[0m ms; h-index computation costs \x1b[1;31m%f\x1b[0m ms \n",
                   initialzation*1000,
                   find_outcore*1000,
                   h_index_computation*1000);

        }
    }
    else{
        //for all the (k,0)-cores with 0 <= k <= N, we maintain the l_{max}(v, k) value of vertices
        //in the DFS search based way given edge deletion.
        if(reuse_pruning && k0core_pruning){
            vector<vector<ArrayEntry>> k_adj_in_cur(n_), k_adj_out_cur(n_);
            vector<uint32_t> mED_out_cur(n_, 0);

            //first get the unique (k, 0)-cores
            vector<int> unique_k0_cores;
            set<int, greater<int>> k0_value_set;
            for (uint32_t i = 0; i < n_; ++i) {
                if(k_max[i] <= M_ ){
                    k0_value_set.insert(k_max[i]);
                }
            }
            for(int i : k0_value_set){
                unique_k0_cores.push_back(i);
            }
            printf("k0core pruning efficiency: %f, %d, %d\n", (float)unique_k0_cores.size() / (float)(M_ + 1), unique_k0_cores.size(), M_ + 1);

             //record the k_max value of edges
            vector<vector<int>> k_max_edges_group(M_ + 1);
            for(uint32_t eid = 0; eid < edges_.size(); ++eid){
                k_max_edges_group[min(M_ , min(k_max[edges_[eid].first], k_max[edges_[eid].second]))].push_back(eid);
            }

             //record the k_max value of modified edges
            vector<vector<int>> k_max_modify_edges_group(M_ + 1);
            for(uint32_t eid = 0; eid < modified_edges.size(); ++eid){
                k_max_modify_edges_group[min(M_ , min(k_max[modified_edges[eid].first], k_max[modified_edges[eid].second]))].push_back(eid);
            }

            for(uint32_t k : unique_k0_cores){
                for (uint32_t eid : k_max_edges_group[k]) {
                    const uint32_t v1 = edges_[eid].first;
                    const uint32_t v2 = edges_[eid].second;
                    k_adj_in_cur[v2].push_back({v1, eid});
                    k_adj_out_cur[v1].push_back({v2, eid});
                    if(l_max[v2][k] >= l_max[v1][k]) {
                        ++mED_out_cur[v1];
                    }
                }

                /*find out-core of inserted edges*/
                vector<bool> compute(n_, false);  //needs to be computed
                vector<bool> be_in_outcore(n_, false);

                #pragma omp parallel for num_threads(lmax_number_of_threads)
                for(auto &eid : k_max_modify_edges_group[k]){
                    auto edge = modified_edges[eid];
                    uint32_t root = edge.first;
                    ASSERT_MSG(k < l_max[edge.second].size() && k < l_max[edge.first].size(),
                               "marker 1 " << k << " " <<  l_max[edge.second].size() << " " << l_max[edge.first].size());
                    if (l_max[edge.second][k] < l_max[edge.first][k]) {
                        root = edge.second;
                    }
                    uint32_t k_M_ = l_max[root][k];
                    for(uint32_t vid = 0; vid < n_; ++vid){
                        if(k_max[vid] >= k && l_max[vid][k] == k_M_ && mED_out_cur[vid] < l_max[vid][k]){
                            ASSERT_MSG(k < l_max[vid].size(),
                                       "marker 2 " << k << " " <<  l_max[vid].size());
                            compute[vid] = true;
                            l_max[vid][k] = k_adj_out_cur[vid].size();
                        }
                    }
                }

                bool flag = true;
                uint32_t round_cnt = 0;
                while (flag){
                    flag = false;
                    #pragma omp parallel for num_threads(lmax_number_of_threads)
                    for(uint32_t vid = 0; vid < n_; ++vid){
                        if(compute[vid]){
                            vector<uint32_t> tmp_neighbor_out_coreness(k_adj_out_cur[vid].size(),0);
                            for(uint32_t i = 0; i < k_adj_out_cur[vid].size(); ++i){
                                ASSERT_MSG(i < tmp_neighbor_out_coreness.size(),
                                           "i >= tmp size " << i << " " << tmp_neighbor_out_coreness.size());
                                ASSERT_MSG(k < l_max[k_adj_out_cur[vid][i].vid].size() && k < l_max[vid].size(),
                                           k <<  " " << l_max[k_adj_out_cur[vid][i].vid].size() << " " <<  l_max[vid].size() << " " << k_max[vid]);

                                tmp_neighbor_out_coreness[i] = l_max[k_adj_out_cur[vid][i].vid][k];
                            }
                            uint32_t tmp_h_index = cal_hIndex(tmp_neighbor_out_coreness);
                            if(tmp_h_index < l_max[vid][k]){
                                l_max[vid][k] = tmp_h_index;
                                flag = true;
                            }
                        }
                    }
                    round_cnt++;
                }
                fill(mED_out_cur.begin(), mED_out_cur.end(), 0);  //reset all values to 0
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
            vector<vector<ArrayEntry>> k_adj_in_cur(n_), k_adj_out_cur(n_);
            vector<uint32_t> mED_out_cur(n_, 0);

            //record the k_max value of edges
            vector<vector<int>> k_max_edges_group(M_ + 1);
            for(uint32_t eid = 0; eid < edges_.size(); ++eid){
                k_max_edges_group[min(M_ , min(k_max[edges_[eid].first], k_max[edges_[eid].second]))].push_back(eid);
            }

            //record the k_max value of modified edges
            vector<vector<int>> k_max_modify_edges_group(M_ + 1);
            for(uint32_t eid = 0; eid < modified_edges.size(); ++eid){
                k_max_modify_edges_group[min(M_ , min(k_max[modified_edges[eid].first], k_max[modified_edges[eid].second]))].push_back(eid);
            }


            for(int k = M_; k >= 0 ; --k) {
                //vector<uint32_t> sub_out_coreness(n_, 0);  //the l_{max} value of vertices in the current (k,0)-core
                for (uint32_t eid : k_max_edges_group[k]) {
                    const uint32_t v1 = edges_[eid].first;
                    const uint32_t v2 = edges_[eid].second;
                    k_adj_in_cur[v2].push_back({v1, eid});
                    k_adj_out_cur[v1].push_back({v2, eid});
                    if(l_max[v2][k] >= l_max[v1][k]) {
                        ++mED_out_cur[v1];
                    }
                }

                /*find out-core of inserted edges*/
                vector<bool> compute(n_, false);  //needs to be computed
                vector<bool> be_in_outcore(n_, false);

                #pragma omp parallel for num_threads(lmax_number_of_threads)
                for(auto &eid : k_max_modify_edges_group[k]){
                    auto edge = modified_edges[eid];
                    uint32_t root = edge.first;
                    ASSERT_MSG(k < l_max[edge.second].size() && k < l_max[edge.first].size(),
                               "marker 1 " << k << " " <<  l_max[edge.second].size() << " " << l_max[edge.first].size());
                    if (l_max[edge.second][k] < l_max[edge.first][k]) {
                        root = edge.second;
                    }
                    uint32_t k_M_ = l_max[root][k];
                    for(uint32_t vid = 0; vid < n_; ++vid){
                        if(k_max[vid] >= k && l_max[vid][k] == k_M_ && mED_out_cur[vid] < l_max[vid][k]){
                            ASSERT_MSG(k < l_max[vid].size(),
                                       "marker 2 " << k << " " <<  l_max[vid].size());
                            compute[vid] = true;
                            l_max[vid][k] = k_adj_out_cur[vid].size();
                        }
                    }
                }

                bool flag = true;
                uint32_t round_cnt = 0;
                while (flag){
                    flag = false;
                    #pragma omp parallel for num_threads(lmax_number_of_threads)
                    for(uint32_t vid = 0; vid < n_; ++vid){
                        if(compute[vid]){
                            vector<uint32_t> tmp_neighbor_out_coreness(k_adj_out_cur[vid].size(),0);
                            for(uint32_t i = 0; i < k_adj_out_cur[vid].size(); ++i){
                                ASSERT_MSG(i < tmp_neighbor_out_coreness.size(),
                                           "i >= tmp size " << i << " " << tmp_neighbor_out_coreness.size());
                                ASSERT_MSG(k < l_max[k_adj_out_cur[vid][i].vid].size() && k < l_max[vid].size(),
                                           k <<  " " << l_max[k_adj_out_cur[vid][i].vid].size() << " " <<  l_max[vid].size() << " " << k_max[vid]);

                                tmp_neighbor_out_coreness[i] = l_max[k_adj_out_cur[vid][i].vid][k];
                            }
                            uint32_t tmp_h_index = cal_hIndex(tmp_neighbor_out_coreness);
                            if(tmp_h_index < l_max[vid][k]){
                                l_max[vid][k] = tmp_h_index;
                                flag = true;
                            }
                        }
                    }
                    round_cnt++;
                }
                fill(mED_out_cur.begin(), mED_out_cur.end(), 0);  //reset all values to 0
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
            printf("k0core pruning efficiency: %f, %d, %d\n", (float)unique_k0_cores.size() / (float)(M_ + 1), unique_k0_cores.size(), M_ + 1);


            for(uint32_t k : unique_k0_cores) {
                vector<vector<ArrayEntry>> k_adj_in(n_), k_adj_out(n_);
                vector<uint32_t> mED_out(n_, 0);
                //vector<bool> is_in_subgraph (n_, false);   //wether the vertex is in the current (k,0)-core
                vector<uint32_t> sub_out_coreness(n_, 0);  //the l_{max} value of vertices in the current (k,0)-core
                for (uint32_t eid = 0; eid < edges_.size(); ++eid) {
                    const uint32_t v1 = edges_[eid].first;
                    const uint32_t v2 = edges_[eid].second;
                    if(k_max[v1] >= k && k_max[v2] >= k){
                        k_adj_in[v2].push_back({v1, eid});
                        k_adj_out[v1].push_back({v2, eid});
                        if(l_max[v2][k] >= l_max[v1][k]){
                            ++mED_out[v1];
                        }
                    }
                }


                /*find out-core of inserted edges*/
                vector<bool> compute(n_, false);  //needs to be computed
                vector<bool> be_in_outcore(n_, false);
                /*find out-core of inserted edges*/


                #pragma omp parallel for num_threads(lmax_number_of_threads)
                for(auto & edge : modified_edges){
                    if(k_max[edge.first] >= k && k_max[edge.second] >= k){
                        uint32_t root = edge.first;
                        ASSERT_MSG(k < l_max[edge.second].size() && k < l_max[edge.first].size(), 
                                    "marker 1 " << k << " " <<  l_max[edge.second].size() << " " << l_max[edge.first].size());
                        if (l_max[edge.second][k] < l_max[edge.first][k]) {
                            root = edge.second;
                        }

                        uint32_t k_M_ = l_max[root][k];

                        for(uint32_t vid = 0; vid < n_; ++vid){
                            if(k_max[vid] >= k && l_max[vid][k] == k_M_ && mED_out[vid] < l_max[vid][k]){
                                ASSERT_MSG(k < l_max[vid].size(), 
                                    "marker 2 " << k << " " <<  l_max[vid].size());
                                compute[vid] = true;
                                l_max[vid][k] = k_adj_out[vid].size();
                            }
                        }
                    }
                }




                bool flag = true;
                uint32_t round_cnt = 0;
                while (flag){
                    flag = false;
                    #pragma omp parallel for num_threads(lmax_number_of_threads)
                    for(uint32_t vid = 0; vid < n_; ++vid){
                        if(compute[vid]){
                            vector<uint32_t> tmp_neighbor_out_coreness(k_adj_out[vid].size(),0);
                            for(uint32_t i = 0; i < k_adj_out[vid].size(); ++i){
                                ASSERT_MSG(i < tmp_neighbor_out_coreness.size(), 
                                            "i >= tmp size " << i << " " << tmp_neighbor_out_coreness.size());
                                ASSERT_MSG(k < l_max[k_adj_out[vid][i].vid].size() && k < l_max[vid].size(), 
                                            k <<  " " << l_max[k_adj_out[vid][i].vid].size() << " " <<  l_max[vid].size() << " " << k_max[vid]);

                                tmp_neighbor_out_coreness[i] = l_max[k_adj_out[vid][i].vid][k];
                            }
                            uint32_t tmp_h_index = cal_hIndex(tmp_neighbor_out_coreness);
                            if(tmp_h_index < l_max[vid][k]){
                                l_max[vid][k] = tmp_h_index;
                                flag = true;
                            }
                        }
                    }
                    round_cnt++;
                }
            }
            printf("deletion l_max main calculation done\n");

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
        else{
            for(uint32_t k = 0; k <= M_ ; ++k) {
                vector<vector<ArrayEntry>> k_adj_in(n_), k_adj_out(n_);
                vector<uint32_t> mED_out(n_, 0);
                //vector<bool> is_in_subgraph (n_, false);   //wether the vertex is in the current (k,0)-core
                vector<uint32_t> sub_out_coreness(n_, 0);  //the l_{max} value of vertices in the current (k,0)-core
                for (uint32_t eid = 0; eid < edges_.size(); ++eid) {
                    const uint32_t v1 = edges_[eid].first;
                    const uint32_t v2 = edges_[eid].second;
                    if(k_max[v1] >= k && k_max[v2] >= k){
                        k_adj_in[v2].push_back({v1, eid});
                        k_adj_out[v1].push_back({v2, eid});
                        if(l_max[v2][k] >= l_max[v1][k]){
                            ++mED_out[v1];
                        }
                    }
                }


                /*find out-core of inserted edges*/
                vector<bool> compute(n_, false);  //needs to be computed
                vector<bool> be_in_outcore(n_, false);
                /*find out-core of inserted edges*/


                #pragma omp parallel for num_threads(lmax_number_of_threads)
                for(auto & edge : modified_edges){
                    if(k_max[edge.first] >= k && k_max[edge.second] >= k){
                        uint32_t root = edge.first;
                        ASSERT_MSG(k < l_max[edge.second].size() && k < l_max[edge.first].size(), 
                                    "marker 1 " << k << " " <<  l_max[edge.second].size() << " " << l_max[edge.first].size());
                        if (l_max[edge.second][k] < l_max[edge.first][k]) {
                            root = edge.second;
                        }

                        uint32_t k_M_ = l_max[root][k];

                        for(uint32_t vid = 0; vid < n_; ++vid){
                            if(k_max[vid] >= k && l_max[vid][k] == k_M_ && mED_out[vid] < l_max[vid][k]){
                                ASSERT_MSG(k < l_max[vid].size(), 
                                    "marker 2 " << k << " " <<  l_max[vid].size());
                                compute[vid] = true;
                                l_max[vid][k] = k_adj_out[vid].size();
                            }
                        }
                    }
                }




                bool flag = true;
                uint32_t round_cnt = 0;
                while (flag){
                    flag = false;
                    #pragma omp parallel for num_threads(lmax_number_of_threads)
                    for(uint32_t vid = 0; vid < n_; ++vid){
                        if(compute[vid]){
                            vector<uint32_t> tmp_neighbor_out_coreness(k_adj_out[vid].size(),0);
                            for(uint32_t i = 0; i < k_adj_out[vid].size(); ++i){
                                ASSERT_MSG(i < tmp_neighbor_out_coreness.size(), 
                                            "i >= tmp size " << i << " " << tmp_neighbor_out_coreness.size());
                                ASSERT_MSG(k < l_max[k_adj_out[vid][i].vid].size() && k < l_max[vid].size(), 
                                            k <<  " " << l_max[k_adj_out[vid][i].vid].size() << " " <<  l_max[vid].size() << " " << k_max[vid]);

                                tmp_neighbor_out_coreness[i] = l_max[k_adj_out[vid][i].vid][k];
                            }
                            uint32_t tmp_h_index = cal_hIndex(tmp_neighbor_out_coreness);
                            if(tmp_h_index < l_max[vid][k]){
                                l_max[vid][k] = tmp_h_index;
                                flag = true;
                            }
                        }
                    }
                    round_cnt++;
                }
            }
        }
    }
}

