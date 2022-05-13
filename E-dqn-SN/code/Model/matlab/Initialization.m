function [X,minInbalance,minIndex,partitionNum] = Initialization(G,adjMatrix,nodeNum,edgeNum,iterNum)
%% initialization
    X = zeros(iterNum,nodeNum);
    for i=1:iterNum
        for j=1:nodeNum
            X(i,j) = j;
        end
    end

%% 随机顺序抽取边，按边顺序进行合并
%    disp("start random mergence");
    for i = 1:iterNum
        edge_seq = randperm(edgeNum);   %乱序抽取边
        for j = 1:edgeNum
            [s,t]=findedge(G,edge_seq(j));
            if adjMatrix(s,t) == 1 && X(i,s) ~= X(i,t)
               X(i,:) = Merge(X(i,:), nodeNum, X(i,s), X(i,t));
            end
        end
    end
%    disp("random mergence finish");

%% 确保所有结果合并为2组
    for i = 1:iterNum
        X(i,:) = Merge2two(G,adjMatrix,X(i,:),edgeNum,nodeNum);
    end
%	disp("已将所有边合并为两组");

%% 计算不平衡边最少的方法
%    disp("开始计算不平衡边最少的方案");
    minInbalance = Inbalance(G,adjMatrix,X(1,:),edgeNum);
    minIndex = 1;
    for i = 2:iterNum
        result = Inbalance(G,adjMatrix,X(i,:),edgeNum);
        if(result < minInbalance)
            minInbalance = result;
            minIndex = i;
        end
    end
%    disp("结束");
    %partitionNum = PartitionsNum(X(minIndex,:),nodeNum);
    partitionNum = 2;

end

%% 合并两组
function result = Merge(X,nodeNum,s_value,t_value)
    for i = 1:nodeNum
        if X(i) == s_value
            X(i) = t_value;
        end
    end
    result = X;
end

%% 计算不平衡边数
function result = Inbalance(G,adjMatrix,X,edgeNum)
	result = 0; %sum of number of positive links between different clusters and negative links within the same cluster
    for  i=1:edgeNum
        [s,t]=findedge(G,i);
        if((adjMatrix(s,t)<0)&&(X(s)==X(t)))
            %negative links within a cluster
            result = result + 1;
        else
            if((adjMatrix(s,t)>0)&&(X(s)~=X(t)))
                % positive links between clusters
                result = result + 1;
            end
        end
    end
end

%% 计算分组数量
function [groupNum,groups] = PartitionsNum(X,nodeNum)
    groupNum = 0;
    groups = [];
    flag = zeros(nodeNum);
    for i = 1:nodeNum
        if(flag(X(i)) == 0)
            flag(X(i))= flag(X(i)) + 1;
            groups = [groups, X(i)];
        end
    end
	for i = 1:nodeNum
        if(flag(i) ~= 0)
            groupNum = groupNum + 1;
        end
	end
end

%% 计算最佳合并项
function [s_group,t_group] = bestMergeGroup(G,adjMatrix,X,groups,groupNum,nodeNum,edgeNum)
    s_group = groups(1);
	t_group = groups(1);
    mindelta = edgeNum;
	for i = 1:groupNum-1
        if(mod(i,100) == 0)
            disp("i=");
            disp(i);
        end
        for j = i+1:groupNum
            if i == j
                continue;
            end
            delta = InbalanceChange(adjMatrix, X, groups(i), groups(j), nodeNum);
            if (delta < mindelta)
                mindelta = delta;
                s_group = groups(i);
                t_group = groups(j);
            end
        end
	end
end
%% 把结果合并为2组
function X = Merge2two(G,adjMatrix,X,edgeNum,nodeNum)
    [groupNum,groups] = PartitionsNum(X,nodeNum);
    while groupNum > 2
        % 贪心选择最合适的组进行合并
    	%[s_group,t_group] = bestMergeGroup(G,adjMatrix,X,groups,groupNum,nodeNum,edgeNum);
    	% 随机合并
    	[s_group,t_group] = randMergeGourp(groups,groupNum);
        X = Merge(X,nodeNum,s_group,t_group);
        [groupNum,groups] = PartitionsNum(X,nodeNum);
    end
    value0 = groups(1);
    value1 = groups(2);
    for i = 1:nodeNum
        if(X(i) == value0)
            X(i) = 0;
        end
        if(X(i) == value1)
            X(i) = 1;
        end
    end
end

%% 计算两个组合并后的不平衡边变化情况
function delta = InbalanceChange(adjMatrix, X, s, t, nodeNum)
    s_index = [];
    t_index = [];
    sNum = 0;
    tNum = 0;
    delta = 0;
    for i = 1:nodeNum
       if X(i) == s
          s_index = [s_index,i];
          sNum = sNum + 1;
       end
    end
    for i = 1:nodeNum
       if X(i) == t
          t_index = [t_index,i];
          tNum = tNum + 1;
       end
    end
    for i = 1:sNum
        for j = 1:tNum
            sign = adjMatrix(s_index(i),t_index(j));
            if(sign == -1)
               delta = delta + 1;
            else
                if(sign == 1)
                   delta = delta - 1;
                end
            end
        end
    end
end

%% 随机选择两个组进行合并
function [s_group,t_group] = randMergeGourp(groups,groupNum)
    s_index = randi(groupNum,1,1);
    t_index = randi(groupNum,1,1);
    while(s_index == t_index)
        t_index = randi(groupNum,1,1);
    end
    s_group = groups(s_index);
    t_group = groups(t_index);
end
