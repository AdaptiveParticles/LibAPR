function analysis_data = load_analysis_data(name,varargin)
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


%now add all the attributes attached to the file
for i = 1:length(file_info.GroupHierarchy(1).Groups(1).Attributes)
    full_path = file_info.GroupHierarchy(1).Groups(1).Attributes(i).Name;
    
    name_index = strfind(full_path,'/');
    name_index = name_index(end);
    
    data_name = full_path((name_index+1):end);
    
    dset_id = H5A.open(group_id,data_name);
    
    analysis_data.(data_name) = H5A.read(dset_id);
    
    H5A.close(dset_id);
end

%now the data
%now add all the datasets attached to the file
for i = 1:length(file_info.GroupHierarchy(1).Groups(1).Datasets)
    full_path = file_info.GroupHierarchy(1).Groups(1).Datasets(i).Name;
    
    name_index = strfind(full_path,'/');
    name_index = name_index(end);
    
    data_name = full_path((name_index+1):end);
    
    dset_id = H5D.open(fid,full_path);
    
    index = strfind(data_name,'.');
    
    if(~isempty(index))
        data_name(index) = '0'; 
    end
    
    index = isspace(data_name);
    
    if(~isempty(index))
        data_name(index) = 0; 
    end
    
    if(strfind(data_name,':'))
        data_name = data_name(1:(end-1));
    end
    
    
    
    try
        analysis_data.(data_name) = double(H5D.read(dset_id));
        H5D.close(dset_id)
    catch
        
        index = isletter(data_name);
        index = ~index;
        
        if(~isempty(index))
            data_name(index) = '_';
        end
        
        analysis_data.(data_name) = double(H5D.read(dset_id));
        H5D.close(dset_id)
        
    end
end



end


