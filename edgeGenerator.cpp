#include "edgeGenerator.h"

/**
 * @brief Construct a new edge Generator::edge Generator object
 * @param graph
 */
edgeGenerator::edgeGenerator(const std::vector<std::pair<uint32_t, uint32_t>>& graph)
{
    m_graph = graph;
    std::random_device rd;
    m_engine = std::default_random_engine(rd());

    for(auto it : m_graph){
        node_recorder.insert(it.first);
        node_recorder.insert(it.second);
    }

    uint32_t vertex_id = 0;
    for(uint32_t tmp_id : node_recorder){
        if(tmp_id != vertex_id){
            ASSERT_MSG(tmp_id == vertex_id, "vertex id is not continuous");
        }
        vertex_id++;
    }

    m_uiN = node_recorder.size();
}

edgeGenerator::~edgeGenerator() {}

/**
 * @brief After edge deletion, some vertices may be deleted as well, leading to inconsistent node id.
 *        return the map from new zero-based node id after deletion to old node id
 */
//std::map<::uint32_t,::uint32_t> edgeGenerator::getMap(std::map<::uint32_t,::uint32_t> &nodeMap) const{
//    return nodeMap;
//}

/**
 * @brief check whether the edge is valid
 * @param: a generated edge
 */
bool edgeGenerator::isValidEdge(const std::pair<uint32_t, uint32_t>& edge) const {
    // An edge is valid if the vertices it connects exist in the original graph
    uint32_t u = edge.first;
    uint32_t v = edge.second;

    return isValidVertex(u) && isValidVertex(v);
}
/**
 * @brief check whether the vertex is valid
 * @param v
 */
bool edgeGenerator::isValidVertex(uint32_t v) const {
    // A vertex is valid if it is smaller than the size of the original graph
    return v >= 0 && v < m_uiN;
}
/**
 * @brief generate random inserting edges that do not already exist in the original graph and whose endpoints exist in the graph
 * @param numEdges, output file path
 */
void edgeGenerator::generatorInsertEdges(uint32_t numEdges, char *pcFile) {
    std::vector<std::pair<uint32_t, uint32_t>> graph_after_insertion(m_graph); //to avoid duplicate inserted edges
    std::uniform_int_distribution<uint32_t> dist(0, m_uiN - 1);
    std::vector<std::pair<uint32_t, uint32_t>> generated_edges;

    //first calculate the in-degree and out-degree of vertices,
    // to avoid generating edges that will lead to isolated vertices
    vector<int> in_degree(m_uiN,0), out_degree(m_uiN,0);
    for(auto edge : m_graph){
        in_degree[edge.second]++;
        out_degree[edge.first]++;
    }

    while (generated_edges.size() < numEdges) {
        uint32_t u = dist(m_engine);
        uint32_t v = dist(m_engine);
        ASSERT_MSG( 0 < u < m_uiN &&  0 < v < m_uiN, "vertex id out of range");

        if (u != v && std::find(graph_after_insertion.begin(), graph_after_insertion.end(), std::make_pair(u, v)) == graph_after_insertion.end()
        && isValidEdge(std::make_pair(u, v)) && out_degree[u] > 0 && in_degree[v] > 0
        && out_degree[v] > 0 && in_degree[u] > 0 ) {
            graph_after_insertion.emplace_back(u, v); //to avoid duplicate inserted edges
            generated_edges.emplace_back(u, v);
            in_degree[v]++;
            out_degree[u]++;
        }
    }

    set<uint32_t> node_set;
    for(auto e : graph_after_insertion){
        node_set.insert(e.first);
        node_set.insert(e.second);
    }
    ASSERT_MSG(node_set.size() == m_uiN, "vertex id after insert edge generation is not continuous");
   
    // vector<GeneratedEdgeEntry> result (generated_edges.size());
    // for(int i = 0; i < generated_edges.size(); ++i){
    //     result[i].u = generated_edges[i].first;
    //     result[i].v = generated_edges[i].second;
    //     result[i].smaller_in_degree = min(in_degree[generated_edges[i].first], in_degree[generated_edges[i].second]);
    // }
    // sort(result.begin(), result.end(), in_degree_cmp);


    sort(generated_edges.begin(),generated_edges.end());
    FILE* fp = fopen (pcFile, "w+");
    ASSERT_MSG(NULL != fp, "invalid output file");
    for(const auto & it : generated_edges/*result*/)
    {
        fprintf(fp, "%d %d\n", it.first/*u*/, it.second/*v*/);
    }

    fclose(fp);

}


/**
 * @brief generate random deleting edges that exist in the original graph
 * @param numEdges, output file path
 */
void edgeGenerator::generatorDeleteEdges(uint32_t numEdges, char *pcFile) {
    std::uniform_int_distribution<uint32_t> dist(0, m_graph.size() - 1);
    std::vector<std::pair<uint32_t, uint32_t>> generated_edges; //edges to be deleted
    std::vector<bool> deleted(m_graph.size(),false); //whether an edge is deleted, to avoid duplicate deleted edges

    //first calculate the in-degree and out-degree of vertices,
    // to avoid generating edges that will lead to isolated vertices
    vector<uint32_t> in_degree(m_uiN,0), out_degree(m_uiN,0);
    for(auto edge : m_graph){
        in_degree[edge.second]++;
        out_degree[edge.first]++;
    }

    while (generated_edges.size() < numEdges) {
        uint32_t index = dist(m_engine);
        if(!deleted[index]){
            auto edge = m_graph[index];
            if (isValidEdge(edge) && out_degree[edge.first] > 1 && in_degree[edge.second] > 1
                && in_degree[edge.first] > 1 && out_degree[edge.second] > 1) {
                deleted[index] = true;
                generated_edges.push_back(edge);
                --in_degree[edge.second];
                --out_degree[edge.first];
            }
        }
    }

    std::vector<std::pair<uint32_t, uint32_t>> graph_after_deletion;
    for(uint32_t  i = 0; i < m_graph.size(); i++){
        if(!deleted[i]){
            graph_after_deletion.push_back(m_graph[i]);
        }
    }
    set<uint32_t> node_set;
    for(auto e : graph_after_deletion){
        node_set.insert(e.first);
        node_set.insert(e.second);
    }
    ASSERT_MSG(node_set.size() == m_uiN, "vertex id after insert edge generation is not continuous");
    ASSERT_MSG(generated_edges.size() == numEdges, "generated edges size is not equal to numEdges");

    // vector<GeneratedEdgeEntry> result (generated_edges.size());
    // for(int i = 0; i < generated_edges.size(); ++i){
    //     result[i].u = generated_edges[i].first;
    //     result[i].v = generated_edges[i].second;
    //     result[i].smaller_in_degree = min(in_degree[generated_edges[i].first], in_degree[generated_edges[i].second]);
    // }
    // sort(result.begin(), result.end(), in_degree_cmp);


    sort(generated_edges.begin(),generated_edges.end());
    FILE* fp = fopen (pcFile, "w+");
    ASSERT_MSG(NULL != fp, "invalid output file");
    for(const auto & it : generated_edges/*result*/)
    {
        fprintf(fp, "%d %d\n", it.first/*u*/, it.second/*v*/);
    }

    fclose(fp);
}

/**
 * @brief generate subgraphs based on the input graph, result in subgraph with 20%/40%/60%/80% edges
 * @param input subgraph, output file path
 */
void edgeGenerator::generatorSubGraphs(vector<pair<uint32_t, uint32_t>> input_graph, string pcFile) {
    vector<string> subgraph_file_suffix = {"-0.2.txt","-0.4.txt","-0.6.txt","-0.8.txt"};
    vector<double> subgraph_ratio = {0.2, 0.4, 0.6, 0.8};
    ASSERT_MSG(subgraph_file_suffix.size() == subgraph_ratio.size(), "subgraph file suffix size is not equal to subgraph ratio size");

    for(int i = 0; i < subgraph_ratio.size();i++){
        /*obtain the output file path*/
        string suffix = subgraph_file_suffix[i];
        string output_subfile_string= pcFile + suffix;
        char *output_file = (char*)output_subfile_string.c_str();

        uint32_t numEdges = subgraph_ratio[i] * input_graph.size();
        std::uniform_int_distribution<uint32_t> dist(0, numEdges - 1);
        vector<pair<uint32_t, uint32_t>> generated_edges;
        vector<bool> edge_exsits(numEdges, false);

        while (generated_edges.size() < numEdges) {
            uint32_t index = dist(m_engine);
            if(!edge_exsits[index]){
                edge_exsits[index] = true;
                generated_edges.push_back(input_graph[index]);
            }
        }

        sort(generated_edges.begin(),generated_edges.end());

        FILE* fp = fopen (output_file, "w+");
        ASSERT_MSG(NULL != fp, "invalid output file");
        for(const auto & it : generated_edges)
        {
            fprintf(fp, "%d %d\n", it.first, it.second);
        }
        printf("generate %f done\n", subgraph_ratio[i]);

        fclose(fp);
    }







}

/**
 * @brief get the edge batch for parallel processing with the edges to be inserted/deleted
 * @return
 */
std::vector<vector<std::pair<uint32_t, uint32_t>>> edgeGenerator::getEdgeBatch(
        const std::vector<std::pair<uint32_t, uint32_t>> &edges_to_be_modified,
        const  vector<vector<pair<::uint32_t,::uint32_t>>> & old_d_core_decomposition,
        vector<pair<::uint32_t,::uint32_t>> &remaining_unbatched_edges,
        const bool kmax_hierarchy, const bool kedge_set) {
    auto test1 = omp_get_wtime();

    remaining_unbatched_edges.clear();
    vector<vector<pair<uint32_t, uint32_t>>> generated_edge_batch;
    /*initializations*/
    uint32_t max_k_max = 0, max_vertex_id = 0;     //maximal edge k_max value, maximal vertex id in the inserted/deleted edges
    for(const auto &e: edges_to_be_modified){
            max_vertex_id = std::max(max_vertex_id, std::max(e.first, e.second));
    }
    uint32_t vec_bound = max_vertex_id + 1;
    auto test2 = omp_get_wtime();

    vector<ArrayEntry> edge_k_max(edges_to_be_modified.size()); //k_max value of inserted/deleted edges
    #pragma omp parallel for num_threads(32)
    for(int eid = 0; eid < edges_to_be_modified.size(); ++eid){
        auto e = edges_to_be_modified[eid];
        if(old_d_core_decomposition[e.first][0].first < old_d_core_decomposition[e.second][0].first){
            edge_k_max[eid].vid = e.first;
            edge_k_max[eid].kmax = old_d_core_decomposition[e.first][0].first;
        }
        else{
            edge_k_max[eid].vid = e.second;
            edge_k_max[eid].kmax = old_d_core_decomposition[e.second][0].first;
        }
    }

    for(int eid = 0; eid < edges_to_be_modified.size(); ++eid){
        max_k_max = max(max_k_max, edge_k_max[eid].kmax);
    }
    auto test3 = omp_get_wtime();

    vector<vector<int>> B(max_k_max + 1); // Empty buckets
    for (int eid = 0; eid < edges_to_be_modified.size(); ++eid) {
        B[edge_k_max[eid].kmax].push_back(eid);
    }

    //remove empty buckets
    B.erase (std::remove_if (B.begin (), B.end (), [] (const auto& vv)
    {
        return vv.empty ();
    }), B.end ());
    auto test4 = omp_get_wtime();

    vector<vector<pair<uint32_t, uint32_t>>> batches_kmax_hierarchy, batches_kedge_set; //batches of edges,
    uint32_t k_edge_set_size = 0, k_max_hierarchy_size = 0, generated_edge_batch_size = 0;
    if(kedge_set){
        /*process remaining dis-batched B with edges has same k_max value*/
        for(auto &batch : B){        //each batch is a list of edges with same k_max value
            bool flag = true;
            while (flag){
                //vector<int> candidate_batch;
                vector<pair<uint32_t,uint32_t>> candidate_batch_edge;
                vector<bool> v_ (vec_bound, false);
                for(auto it = batch.begin(); it != batch.end(); ){
                    int eid = *(it);
                    if(candidate_batch_edge.empty() || !v_[edge_k_max[eid].vid]) {
                        //candidate_batch.push_back(eid);
                        candidate_batch_edge.push_back(edges_to_be_modified[eid]);
                        v_[edge_k_max[eid].vid] = true;
                        it = batch.erase(it);
                    }
                    else{
                        ++it;
                    }
                }
                uint32_t batch_size = candidate_batch_edge.size();
                if(batch_size > 1) {    //avoid single edge as batch
                    batches_kedge_set.push_back(candidate_batch_edge);
                    k_max_hierarchy_size += candidate_batch_edge.size();
                }
                else if(batch_size == 1){     //single edge as a batch
                    remaining_unbatched_edges.push_back(/*edges_to_be_modified[candidate_batch[0]]*/candidate_batch_edge[0]);   //sequential processing
                    flag = false;
                }
                else{
                    flag = false;
                }
            }
        }
        generated_edge_batch.insert(generated_edge_batch.end(), batches_kedge_set.begin(), batches_kedge_set.end());
        //generated_edge_batch_eid.insert(generated_edge_batch_eid.end(),batches_kedge_set.begin(), batches_kedge_set.end());
    }
    if(kmax_hierarchy){
        while (B.size() > 1) {
            vector<int> S;
            vector<pair<uint32_t, uint32_t>> S_edge;
            for (auto it = B.begin(); it != B.end();) {
                if (!it->empty()) {   //auto it is an edge list
                    if(S.empty()){
                        int eid = *(it->begin());
                        it->erase(it->begin());
                        S.push_back(eid);
                        S_edge.push_back(edges_to_be_modified[eid]);
                    } else {
                        uint32_t abs_diff = abs_diff(edge_k_max[*(it->begin())].kmax, edge_k_max[S[0]].kmax);
                        if ( abs_diff > 1 ) {
                            int eid = *(it->begin());
                            it->erase(it->begin());
                            S.push_back(eid);
                            S_edge.push_back(edges_to_be_modified[eid]);
                        }
                    }
                    ++it;
                }
                else {
                    it = B.erase(it);
                }
            }
            if(S.empty()){  //no edge can be added to S, meaning no edges can be batched based on k_max value
                break;
            }
            batches_kmax_hierarchy.push_back(S_edge);
            k_edge_set_size += S_edge.size();
        }
        generated_edge_batch.insert(generated_edge_batch.end(), batches_kmax_hierarchy.begin(), batches_kmax_hierarchy.end());
    }



    


    printf("batches_kmax_hierarchy.size(): %d, batches_kedge_set.size(): %d \n", batches_kmax_hierarchy.size(),batches_kedge_set.size());
    /*remove empty buckets*/
    B.erase (std::remove_if (B.begin (), B.end (), [] (const auto& vv)
    {
        return vv.empty ();
    }), B.end ());

    for(auto &batch : B){
        if(!batch.empty()){
            for(auto &eid : batch){
                remaining_unbatched_edges.push_back(edges_to_be_modified[eid]);
            }
        }
    }

    auto test5 = omp_get_wtime();

   /**
    * verification
    */
    /*write a code to check whether there are duplicate edges in batches_kmax_hierarchyand batches_kedge_set*/
    // vector<vector<bool>> verification_edge_exisits(vec_bound, vector<bool>(vec_bound, false));
    // for(auto &batch : batches_kmax_hierarchy){
    //     for(auto &e : batch){
    //         ASSERT_MSG(!verification_edge_exisits[e.first][e.second], "duplicate edge in k_max_hierarchy");
    //         verification_edge_exisits[e.first][e.second] = true;
    //     }
    // }
    // for(auto &batch : batches_kedge_set){
    //     for(auto &e : batch){
    //         ASSERT_MSG(!verification_edge_exisits[e.first][e.second], "duplicate edge in k_edge_set");
    //         verification_edge_exisits[e.first][e.second] = true;
    //     }
    // }

    
    // for(const auto& it : batches_kedge_set){
    //     k_edge_set_size += it.size();
    // }
    // for(const auto& it : batches_kmax_hierarchy){
    //     k_max_hierarchy_size += it.size();
    // }
    // for(const auto& it : generated_edge_batch){
    //     generated_edge_batch_size += it.size();
    // }
    // ASSERT_MSG((k_edge_set_size + k_max_hierarchy_size + remaining_unbatched_edges.size()) == edges_to_be_modified.size(), "edge batch size is not equal to edges_to_be_modified size " << k_edge_set_size << " " << k_max_hierarchy_size << " " << remaining_unbatched_edges.size() << " " << edges_to_be_modified.size());
    // ASSERT_MSG(k_edge_set_size + k_max_hierarchy_size == generated_edge_batch_size, "edge batch size is not equal to generated_batch size");
    // for(const auto& it : generated_edge_batch){
    //     ASSERT_MSG(!it.empty(), "empty edge batch");
    // }
    printf("k_max_hierarchy_size: %d, k_edge_set_size: %d,  remaining_unbatched_edges: %d\n", k_max_hierarchy_size, k_edge_set_size,  remaining_unbatched_edges.size());

    auto test6 = omp_get_wtime();

    printf("batch-1: \x1b[1;31m%f\x1b[0m ms; batch-2: \x1b[1;31m%f\x1b[0m ms; batch-3: \x1b[1;31m%f\x1b[0m ms; "
            "batch-4: \x1b[1;31m%f\x1b[0m ms; batch-verify: \x1b[1;31m%f\x1b[0m ms\n",
            (test2 - test1)*1000,
            (test3 - test2)*1000,
            (test4 - test3)*1000,
            (test5 - test4)*1000,
            (test6 - test5)*1000);

    return generated_edge_batch;
}

/**
 * @brief get the graph after edge insertion, returned graph have zero-based, consistent id
 * @return
 */
std::vector<std::pair<uint32_t, uint32_t>> edgeGenerator::getInsertedGraph(){
    // Return a copy of the original graph with the randomly added and deleted edges
    std::vector<std::pair<uint32_t, uint32_t>> result(m_graph);

    // Add the inserted edges
    //printf("m_insert_edges size: %d\n", m_insert_edges.size());
    for (auto edge : m_insert_edges) {
        if (edge.first != -1 && edge.second != -1 && isValidEdge(edge)) {
            result.push_back(edge);
        }
    }


    return result;
}

/**
 * @brief get the graph after edge deletion, returned graph have zero-based, consistent id
 * @return
 */
std::vector<std::pair<uint32_t, uint32_t>> edgeGenerator::getDeletedGraph() {
    // Return a copy of the original graph with the randomly added and deleted edges
    std::vector<std::pair<uint32_t, uint32_t>> result(m_graph);


    // Delete the deleted edges
    for (auto edge : m_delete_edges) {
        if (edge.first != -1 && edge.second != -1 && isValidEdge(edge) /*&& std::find(result.begin(), result.end(), edge) != result.end()*/) {
            result.erase(std::remove(result.begin(), result.end(), edge), result.end());
        }
    }

    // map the node id into zero-based, consistent id
//    set<::uint32_t> node_set;
//    for (auto eid = 0; eid < result.size(); ++eid){
//        const uint32_t x = result[eid].first;
//        const uint32_t y = result[eid].second;
//        node_set.insert(x);
//        node_set.insert(y);
//    }
//    for(auto i = node_set.begin(); i != node_set.end(); ++i){  //build the node map, key: old id, value: new id
//        old_to_new_node_map[*i] = std::distance(node_set.begin(), i);
//        new_to_old_node_map[std::distance(node_set.begin(), i)]= *i;
//    }
//
//    for (auto & eid : result){
//        eid.first = old_to_new_node_map[eid.first];
//        eid.second = old_to_new_node_map[eid.second];
//    }

    return result;
}

/**
 * @brief turn a directed graph into a graph with both edge being a bi-directional edge and output into file
 * @return
 */

void edgeGenerator::getBiGraph(char* pcFile ){
    vector<pair<uint32_t, uint32_t>> vEdges;
    for(auto & edge : m_graph){
        vEdges.push_back(edge);
        vEdges.emplace_back(edge.second, edge.first);
    }
    sort( vEdges.begin(), vEdges.end() );
    vEdges.erase( unique( vEdges.begin(), vEdges.end() ), vEdges.end() );

    FILE* fp = fopen (pcFile, "w+");
    ASSERT_MSG(NULL != fp, "invalid output file");

    for(auto it : vEdges){
        ASSERT_MSG(!(find(vEdges.begin(), vEdges.end(), make_pair(it.second, it.first)) == vEdges.end()),
                   "bi-graph error: " << it.first << " " << it.second);
    }

    for(const auto & it : vEdges)
    {

        fprintf(fp, "%d %d\n", it.first, it.second);

    }
    fclose(fp);
}