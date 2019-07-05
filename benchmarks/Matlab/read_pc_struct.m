function [ip_size,pc_size] = read_pc_struct(name,varargin)
%
%   Bevan Cheeseman 2015
%
%   Reads in the raw particle data structures
%
%


%open read only
fid = H5F.open(name,'H5F_ACC_RDONLY','H5P_DEFAULT');

file_info = hdf5info(name);

%read in the attributes
group_id = H5G.open(fid,file_info.GroupHierarchy(1).Groups(1).Name);

ip_size = [];
pc_size = [];

%now the data
%now add all the datasets attached to the file
for i = 1:length(file_info.GroupHierarchy.Groups.Groups.Datasets)
    full_path = file_info.GroupHierarchy.Groups.Groups.Datasets(i).Name;
    
    name_index = strfind(full_path,'/');
    name_index = name_index(end);
    
    data_name = full_path((name_index+1):end);
    
    dset_id = H5D.open(fid,full_path);
    
    dataset_size = H5D.get_storage_size(dset_id);
    
    index = strfind(data_name,'_');
    
    l_index = isletter(data_name);
    
    if(strcmp('Ip',data_name(l_index)))
       ip_size(end+1) = dataset_size; 
    else
       pc_size(end+1) = dataset_size;  
    end
    
    
    
    
end



end