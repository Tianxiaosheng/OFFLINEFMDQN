#! /usr/bin/env python3

import sys
import deserialization_core.deserialization as deserialization

if __name__ == "__main__":
    deserialization_info = deserialization.Deserialization()
    deserialization_info.get_lon_decision_inputs_by_deserialization(sys.argv[1])
    print(f"deserialization_info.get_lon_decision_inputs_size(): {deserialization_info.get_lon_decision_inputs_size()}")
    if deserialization_info.get_lon_decision_inputs_size() > 0:
        for i in range(deserialization_info.get_lon_decision_inputs_size()):
            deserialization_info.dump_ego_info_by_frame(i)
            deserialization_info.dump_obj_info_by_frame(i)

    else:
        print("No lon decision inputs found")
