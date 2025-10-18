# V3

This folder contains all directories/projects prior to a refactor in October 2025.

Previously, dependencies were managed globally via Poetry,
but since different notebooks sometimes have dependency conflicts,
per-project dependencies were adopted instead.

Directories in the `v3` folder do not necessarily contain a single project,
whereas directories in `v4` clearly distinguish individual projects and enforce a standard structure.