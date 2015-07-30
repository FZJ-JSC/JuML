
file(GLOB CLASSIFICATION_SRC classification/*.cpp)
ADD_LIBRARY(classification ${CLASSIFICATION_SRC})
file(GLOB CLUSTERING_SRC clustering/*.cpp)
ADD_LIBRARY(clustering ${CLUSTERING_SRC})