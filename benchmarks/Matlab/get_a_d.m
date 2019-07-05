function a_d = get_a_d(folder_name,data_name)
%
%   Bevan Cheeseman 2017
%
%   Accumulates multiple data sets together
%
%

files = dir([folder_name,data_name,'*.h5']);

for i = 1:length(files)
    
   a_d_s{i} =  load_analysis_data([folder_name,files(i).name]);
   
   num_runs = length(a_d_s{i}.num_pixels);
   
   a_d_s{i}.max_flow_mesh = [a_d_s{i}.max_flow_mesh;zeros(num_runs-length(a_d_s{i}.max_flow_mesh),1)];
   a_d_s{i}.max_flow_parts = [a_d_s{i}.max_flow_parts;zeros(num_runs-length(a_d_s{i}.max_flow_parts),1)];
   
   a_d_s{i}.part_num_neigh = [a_d_s{i}.part_num_neigh;zeros(num_runs-length(a_d_s{i}.part_num_neigh),1)];
   
   a_d_s{i}.mesh_num_neigh = [a_d_s{i}.mesh_num_neigh;zeros(num_runs-length(a_d_s{i}.mesh_num_neigh),1)];
end




if(~isempty(files))
    
    a_d = a_d_s{1};  
    fields = fieldnames(a_d_s{1});
    
    for i = 1:length(fields)
       for s = 2:length(a_d_s)  
           a_d.(fields{i}) = [a_d_s{s}.(fields{i});a_d.(fields{i})];
       end
    end
    
else 
    a_d = [];
end






end