function idx = merge(Gwl_ud,edgeNum,num_nodes)


disp(['merge beginning ']);

G=graph(Gwl_ud);
% load('test_G.mat'); %dataset
%edgeNum=numedges(G);
%num_nodes = size(Gwl_ud,1);
idx = zeros(1,num_nodes) * 0;
for i =1:num_nodes
    idx(i) = i;
end
edges = [1:1:edgeNum];
edges_rand = randperm(size(edges, 2));


temp = zeros(1,num_nodes) * 0;
temp_scores = zeros(1,num_nodes) * 0;
scores = zeros(1,num_nodes) * 0;

temp_len = 1;
cluster1 = 0;
cluster2 = 0;
score_len = 1;
flag = 0;
len = 1;

for i = 1:edgeNum
     cluster1 = 0;
     cluster2 = 0;
     [s,t]=findedge(G,edges_rand(i));
%      print('s = ',s)
%      print('t = ',t)
     if( Gwl_ud(s,t) >0 )
         for j = 1:len
             if(temp(j) == idx(t))
                 cluster1 = temp(j);
             end
             if(temp(j) == idx(s))
                 cluster2 = temp(j);
             end                
         end
         if(cluster2 ~= 0 && cluster1 == 0)
                flag = 1;
                idx(s) = cluster2;
                idx(t) = cluster2;
         elseif(cluster1 ~= 0 && cluster2 == 0)
                flag = 1;
                idx(s) = cluster1;
                idx(t) = cluster1;
         elseif(cluster1 ~= 0 && cluster2 ~= 0 && cluster2 == cluster1)
                 flag = 1;
         elseif(cluster1 ~= 0 && cluster2 ~= 0 && cluster2 ~= cluster1)
                 flag = 1;
                 for k = 1:num_nodes
                     if(idx(k) == cluster1)
                           idx(k) = cluster2;
                     end
                 end
         end
         if(flag == 0)
            temp(len) = idx(t);
            idx(s) = idx(t);    
            len = len + 1;
         end
         flag = 0 ;
     end
     
end

len = len - 1;
len = 1;
    temp = zeros(1,num_nodes) * 0;  
    for i = 1:num_nodes
        flag = 0;
        for j = 1:len
            if(temp(j) == idx(i))
                flag = 1;
            end
        end
        if(flag == 0)
           temp(len) = idx(i);
           len = len + 1;
        end
    end
    len = len - 1;
for i = 1:num_nodes
    temp_scores(i) = num_nodes + 1;
    scores(i) = num_nodes + 1;
end
cluster_num = len;
%disp(['The len of clusters is ' num2str(len) '.']);
index = [1:1:len - 1];
cluster_class1 = 0;
cluster_class2 = 0;
cluster_score1 = 0;
cluster_score2 = 0;
temp_index = 1; 

score_mix = 100000;

while(cluster_num ~= 2)
    %disp(['The cluster_num of test is ' num2str(cluster_num) '.']);
    for k =1:len            
       temp_scores(k) = 0;   
       scores(k) = 0;
    end
    for i = 1:num_nodes
        for j =1:len            
            if(temp(j) == idx(i))
                temp_scores(j) = temp_scores(j) + 1;
                scores(j) = scores(j) + 1;
            end
            
        end
    end
    scores = sort(scores);
    cluster_score1 = scores(1);
    for i = 1:num_nodes
        if(cluster_score1 == temp_scores(i))
            cluster_class1 = i;
            break;
        end
    end
    scores(1) = num_nodes + 1;
    
    temp_scores(cluster_class1) = num_nodes + 1;
    scores = sort(scores);
    cluster_score2 = scores(1);
    for i = 1:num_nodes
        if(cluster_score2 == temp_scores(i))
            cluster_class2 = i;
            break;
        end
    end
    %idx
    for i =1:num_nodes
        if((idx(i)==temp(cluster_class1) || idx(i)==temp(cluster_class2) ))
            idx(i) = temp(cluster_class1);
        end
    end
    
    len = 1;
    temp = zeros(1,num_nodes) * 0;
    temp_scores = zeros(1,num_nodes) * 0;
    scores = zeros(1,num_nodes) * 0;
    for i = 1:num_nodes
        flag = 0;
        for j = 1:len
            if(temp(j) == idx(i))
                flag = 1;
            end
        end
        if(flag == 0)
           temp(len) = idx(i);
           len = len + 1;
        end
    end
    len = len - 1;
    for i = 1:num_nodes
        temp_scores(i) = num_nodes + 1;
        scores(i) = num_nodes + 1;
    end
    cluster_num = len;
    %temp
    %idx
end
 
fprintf('finish')