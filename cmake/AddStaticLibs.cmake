# Adds staticLib to provided (existing) static target. Useful for merging all dependencies to
# one fat static lib.
# Use: addStaticLibs(libStatic someStaticLibToMerge [evenMoreStaticLibsIfNeeded])

function(addStaticLibs outLibTarget)
    get_target_property(libtype ${outLibTarget} TYPE)
    if(NOT libtype STREQUAL "STATIC_LIBRARY")
        message(FATAL_ERROR "[${outLibTarget}] is not a static lib")
    endif()
    # Get unique list of libraries to be merged
	set(libsToMerge ${ARGV})
	list(REMOVE_AT libsToMerge 0)
	list(REMOVE_DUPLICATES libsToMerge)

	# Get the file names of the libraries to be merged
	foreach(lib ${libsToMerge})
		get_target_property(libtype ${lib} TYPE)
		if(NOT libtype STREQUAL "STATIC_LIBRARY")
			message(FATAL_ERROR "[${lib}] is not a static lib!")
		endif()
		list(APPEND filesToMerge $<TARGET_FILE:${lib}>)
	endforeach()

    set(outLibFile $<TARGET_FILE:${outLibTarget}>)
    # --------------- Mac OS ----------------
    find_program(HAS_LIBTOOL "libtool") #if not found it will go to UNIX section on Mac
	if(APPLE AND HAS_LIBTOOL )
		add_custom_command(TARGET ${outLibTarget} POST_BUILD
				COMMAND libtool -static -o ${outLibFile} ${outLibFile} ${filesToMerge})
    # --------------- UNIX -------------------
	elseif(UNIX)
		message("Linking on UNIX, lol!")
		foreach(lib ${libsToMerge})
			set(libObjDir mergedLibs/${lib}.objDir)

            # create object directory for current lib ...
			add_custom_command(
                    TARGET ${outLibTarget}
                    COMMAND ${CMAKE_COMMAND} -E make_directory ${libObjDir})

            # ... and output all obj from archive, then merge them into outLibTarget
			add_custom_command(
                    TARGET ${outLibTarget} POST_BUILD
					COMMAND ${CMAKE_AR} -x $<TARGET_FILE:${lib}>
                    COMMAND ${CMAKE_AR} rus ${outLibFile} *.o
					WORKING_DIRECTORY ${libObjDir})
		endforeach()
    # --------------- Windows ---------------
	elseif(WIN32)
		message("building on windows!")
		#set_target_properties(${outLibTarget} PROPERTIES STATIC_LIBRARY_FLAGS "${filesToMerge}")
		add_custom_command(TARGET ${outLibTarget} POST_BUILD
			COMMAND lib.exe /OUT:${outLibFile} ${filesToMerge})
	endif()
endfunction()
