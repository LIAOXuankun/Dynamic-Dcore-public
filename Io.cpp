

#include "Io.h"
/**
    return edges in the input graph
**/
void Io::getEdges(vector<pair<uint32_t, uint32_t> > &vDesEdges)
{
    for (auto & it : m_vE)
    {
        vDesEdges.emplace_back(it.x, it.y);
    }
}
/**
    read edges to be inserted / deleted from file
**/
void Io::readModifiedEdges(std::vector<std::pair<uint32_t, uint32_t>> &Edges, char* pcFile){
    Edges.clear();
    FILE* fp = fopen (pcFile, "rt");
    char acBuffer[100];
    ASSERT_MSG(NULL != fp, "invalid graph file");
    uint32_t x = 0;
    uint32_t y = 0;
    while (!feof(fp))
    {
        char *pPos = fgets(acBuffer, 100, fp);
        if (NULL == pPos)
        {
            break;
        }
        int res = sscanf(acBuffer, "%d %d", &x, &y)/*fscanf(acBuffer, "%d %d %d", &x, &y, &t) */;
        ASSERT_MSG(2 == res, "wrong file");

        if(x != y)
        {
            Edges.emplace_back(x, y);
            ASSERT_MSG(x != y, "loops exist in the graph" << x << y);
        }
    }
    fclose(fp);
}

/**
    Handling Io issues, including reading and writing files
**/
void Io::readFromFile(char* pcFile)
{
    // first read the original file, store them in m_vE
    FILE* fp = fopen (pcFile, "rt");
    char acBuffer[100];
    ASSERT_MSG(NULL != fp, "invalid graph file");
    uint32_t x = 0;
    uint32_t y = 0;
    m_uiN = 0;
    m_uiM = 0;
    uint32_t edge_cnt = 0;
    
    while (!feof(fp))
    {
        char *pPos = fgets(acBuffer, 100, fp);
        if (NULL == pPos)
        {
            break;
        }
        int res = sscanf(acBuffer, "%d%d", &x, &y)/*fscanf(acBuffer, "%d %d %d", &x, &y, &t) */;
        ASSERT_MSG(2 == res, "wrong file");
        if(x != y)
        {
            m_vE.push_back({x, y});
            node_counter.insert(x);
            node_counter.insert(y);
            edge_cnt++;
            ASSERT_MSG(x != y, "loops exist in the graph" << x << y);
        }
    }
    fclose(fp);
    m_uiN = node_counter.size();
    m_uiM = m_vE.size();

    uint32_t node_cnt = 0;
    for(auto i = node_counter.begin(); i != node_counter.end(); ++i) {
        node_map[*i] = node_cnt;
        ++node_cnt;
    }

    // for(auto i = node_counter.begin(); i != node_counter.end(); ++i)  //build the node map
    // {
    //     node_map[*i] = std::distance(node_counter.begin(), i);
    // }
    for(auto i = m_vE.begin(); i != m_vE.end(); ++i)  //replace the original vertex id with the continuous id
    {
        i->x = node_map[i->x];
        i->y = node_map[i->y];
    }
}

/**
 *
 * read exsiting d-core decomposition result from file
 */

void Io::readFromFile(char* pcFile, vector<vector<pair<::uint32_t,::uint32_t>>> &d_core_decomposition)
{
    //read the first line: vertex number
    ifstream fin(pcFile);
    uint32_t vertex_num;

    //read remaining lines
    string single_line;
    uint32_t line_number = 0;
    while(getline(fin, single_line)){
        if(line_number == 0){
            vertex_num = stoi(single_line);
            d_core_decomposition.resize(vertex_num);
        }
        else{
            istringstream iss(single_line);
            string single_number;
            vector<string> sinlge_line_split;
            while (getline(iss, single_number, ' ')) {
                sinlge_line_split.push_back(single_number);
            }

            for(uint32_t i = 0; i < sinlge_line_split.size(); ++i){
                d_core_decomposition[line_number - 1].push_back({sinlge_line_split.size() - i - 1,
                                                                 stoi(sinlge_line_split[i])});
            }
        }
        ++line_number;
    }
}

/**
    save data
**/
void Io::writeToFile(char* pcFile, vector<vector<pair<::uint32_t,::uint32_t>>> &d_core_decomposition) const
{
    FILE* fp = fopen (pcFile, "w+");
    ASSERT_MSG(NULL != fp, "invalid output file");

    /*human read mode*/
//    fprintf(fp, "%d\n", d_core_decomposition.size());
//    for(int i = 0 ; i < d_core_decomposition.size(); i++){
//        fprintf(fp, "vertex id: %d: ",i + 1 );
//        //cout << "vertex id " << i+1 << ": ";
//        for(auto & j : d_core_decomposition[i]){
//            fprintf(fp, "[%d %d] ", j.first, j.second);
//            //cout << " [ " << j.first  << " " << j.second << " ] ";
//        }
//        fprintf(fp, "\n");
//        //cout << endl;
//    }

    /*computer read mode*/
    fprintf(fp, "%d\n", d_core_decomposition.size());
    for(int i = 0 ; i < d_core_decomposition.size(); i++){
        for(auto & j : d_core_decomposition[i]){
            fprintf(fp, "%d ", j.second);
        }
        fprintf(fp, "\n");
    }


    fclose(fp);
}
/**
    save data
**/
void Io::writeToFile(char* pcFile, vector<vector<uint32_t>> &l_max) const
{
    FILE* fp = fopen (pcFile, "w+");
    ASSERT_MSG(NULL != fp, "invalid output file");
    fprintf(fp, "%zu\n", l_max.size());
    for(int i = 0 ; i < l_max.size(); ++i){
        fprintf(fp, "vertex id: %d: ",i + 1 );
        for(int j = l_max[i].size() - 1; j >= 0; --j){
            fprintf(fp, "[%d %d] ", j, l_max[i][j]);
        }
        fprintf(fp, "\n");
    }


    fclose(fp);
}


/**
    save data
**/
void Io::writeToFile(char* pcFile, vector<uint32_t> &k_max) const
{
    FILE* fp = fopen (pcFile, "w+");
    ASSERT_MSG(NULL != fp, "invalid output file");
    fprintf(fp, "%zu\n", k_max.size());
    for(int i = 0 ; i < k_max.size(); ++i){
        fprintf(fp, "vertex id: %d: %d ",i + 1, k_max[i]);
        fprintf(fp, "\n");
    }


    fclose(fp);
}

/**
    print decomposition result
**/
void Io::printDecomposition(vector<vector<pair<::uint32_t, ::uint32_t>>> &d_core_decomposition) const {
    cout << "d-core decomposition: " << d_core_decomposition.size() << endl;
    for(int i = 0 ; i < d_core_decomposition.size(); i++){
        cout << "vertex id " << i+1 << ": ";
        for(auto & j : d_core_decomposition[i]){
            cout << " [ " << j.first  << " " << j.second << " ] ";
        }
        cout << endl;
    }
}

/**
    print klists
**/
void Io::printKlists(vector<vector<pair<::uint32_t, ::uint32_t>>> &independent_k_lists) const {
    cout << "independent k-lists: " << independent_k_lists.size() << endl;
    for(int i = 0 ; i < independent_k_lists.size(); i++){
        cout << "k " << independent_k_lists[i][0].first << " size " << independent_k_lists[i].size()-1 << endl;
        for(int j = 1 ; j < independent_k_lists[i].size(); j++){
            cout << independent_k_lists[i][j].first + 1 << " " << independent_k_lists[i][j].second + 1 << " | ";
        }
        cout << endl;
    }
}