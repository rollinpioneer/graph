# Verifier qualification

{
  "status": "PASS",
  "rows": [
    {
      "scenario": "empty_gripper",
      "skill": null,
      "leaked_hidden_into_verifier": false,
      "qa": {
        "GripperEmpty": true,
        "Held:target": false,
        "Held:second_object": false,
        "Open:container": false,
        "Inside:target:container": false,
        "Inside:second_object:container": false,
        "AtBuffer:target:buffer": false,
        "AtBuffer:second_object:buffer": false,
        "OnTable:target": true,
        "OnTable:second_object": true
      },
      "verifier": {
        "p:GripperEmpty": "TRUE",
        "p:Held:target": "FALSE",
        "p:OnTable:target": "TRUE",
        "p:Inside:target:container": "FALSE",
        "p:AtBuffer:target:buffer": "FALSE",
        "p:Held:second_object": "FALSE",
        "p:OnTable:second_object": "TRUE",
        "p:Inside:second_object:container": "FALSE",
        "p:AtBuffer:second_object:buffer": "FALSE",
        "p:Open:container": "FALSE"
      },
      "reasons": {
        "p:GripperEmpty": "cp-disr-d0-verifier-v1:gripper_open_and_no_nearby_blob",
        "p:Held:target": "cp-disr-d0-verifier-v1:open_gripper_or_object_away",
        "p:OnTable:target": "cp-disr-d0-verifier-v1:blob_near_table_plane",
        "p:Inside:target:container": "cp-disr-d0-verifier-v1:xy_inside_container_estimate",
        "p:AtBuffer:target:buffer": "cp-disr-d0-verifier-v1:xy_inside_buffer_estimate",
        "p:Held:second_object": "cp-disr-d0-verifier-v1:open_gripper_or_object_away",
        "p:OnTable:second_object": "cp-disr-d0-verifier-v1:blob_near_table_plane",
        "p:Inside:second_object:container": "cp-disr-d0-verifier-v1:xy_inside_container_estimate",
        "p:AtBuffer:second_object:buffer": "cp-disr-d0-verifier-v1:xy_inside_buffer_estimate",
        "p:Open:container": "cp-disr-d0-verifier-v1:lid_offset_from_container"
      }
    },
    {
      "scenario": "successful_hold",
      "skill": "a:PICK:target:v1",
      "leaked_hidden_into_verifier": false,
      "qa": {
        "GripperEmpty": false,
        "Held:target": true,
        "Held:second_object": false,
        "Open:container": false,
        "Inside:target:container": false,
        "Inside:second_object:container": false,
        "AtBuffer:target:buffer": false,
        "AtBuffer:second_object:buffer": false,
        "OnTable:target": false,
        "OnTable:second_object": true
      },
      "verifier": {
        "p:GripperEmpty": "FALSE",
        "p:Held:target": "TRUE",
        "p:OnTable:target": "FALSE",
        "p:Inside:target:container": "FALSE",
        "p:AtBuffer:target:buffer": "FALSE",
        "p:Held:second_object": "UNKNOWN",
        "p:OnTable:second_object": "TRUE",
        "p:Inside:second_object:container": "FALSE",
        "p:AtBuffer:second_object:buffer": "FALSE",
        "p:Open:container": "FALSE"
      },
      "reasons": {
        "p:GripperEmpty": "cp-disr-d0-verifier-v1:gripper_closed_with_nearby_blob",
        "p:Held:target": "cp-disr-d0-verifier-v1:closed_gripper_near_object_blob",
        "p:OnTable:target": "cp-disr-d0-verifier-v1:object_tracked_at_gripper",
        "p:Inside:target:container": "cp-disr-d0-verifier-v1:xy_inside_container_estimate",
        "p:AtBuffer:target:buffer": "cp-disr-d0-verifier-v1:xy_inside_buffer_estimate",
        "p:Held:second_object": "cp-disr-d0-verifier-v1:held_evidence_conflict",
        "p:OnTable:second_object": "cp-disr-d0-verifier-v1:blob_near_table_plane",
        "p:Inside:second_object:container": "cp-disr-d0-verifier-v1:xy_inside_container_estimate",
        "p:AtBuffer:second_object:buffer": "cp-disr-d0-verifier-v1:xy_inside_buffer_estimate",
        "p:Open:container": "cp-disr-d0-verifier-v1:lid_offset_from_container"
      }
    },
    {
      "scenario": "failed_pick",
      "skill": "a:PICK:target:v1",
      "leaked_hidden_into_verifier": false,
      "qa": {
        "GripperEmpty": true,
        "Held:target": false,
        "Held:second_object": false,
        "Open:container": false,
        "Inside:target:container": false,
        "Inside:second_object:container": false,
        "AtBuffer:target:buffer": false,
        "AtBuffer:second_object:buffer": false,
        "OnTable:target": true,
        "OnTable:second_object": true
      },
      "verifier": {
        "p:GripperEmpty": "UNKNOWN",
        "p:Held:target": "UNKNOWN",
        "p:OnTable:target": "TRUE",
        "p:Inside:target:container": "FALSE",
        "p:AtBuffer:target:buffer": "FALSE",
        "p:Held:second_object": "UNKNOWN",
        "p:OnTable:second_object": "TRUE",
        "p:Inside:second_object:container": "FALSE",
        "p:AtBuffer:second_object:buffer": "FALSE",
        "p:Open:container": "FALSE"
      },
      "reasons": {
        "p:GripperEmpty": "cp-disr-d0-verifier-v1:conflicting_or_missing_gripper_evidence",
        "p:Held:target": "cp-disr-d0-verifier-v1:held_evidence_conflict",
        "p:OnTable:target": "cp-disr-d0-verifier-v1:blob_near_table_plane",
        "p:Inside:target:container": "cp-disr-d0-verifier-v1:xy_inside_container_estimate",
        "p:AtBuffer:target:buffer": "cp-disr-d0-verifier-v1:xy_inside_buffer_estimate",
        "p:Held:second_object": "cp-disr-d0-verifier-v1:held_evidence_conflict",
        "p:OnTable:second_object": "cp-disr-d0-verifier-v1:blob_near_table_plane",
        "p:Inside:second_object:container": "cp-disr-d0-verifier-v1:xy_inside_container_estimate",
        "p:AtBuffer:second_object:buffer": "cp-disr-d0-verifier-v1:xy_inside_buffer_estimate",
        "p:Open:container": "cp-disr-d0-verifier-v1:lid_offset_from_container"
      }
    },
    {
      "scenario": "normal_place",
      "skill": "a:PLACE:target:container:v1",
      "leaked_hidden_into_verifier": false,
      "qa": {
        "GripperEmpty": true,
        "Held:target": false,
        "Held:second_object": false,
        "Open:container": true,
        "Inside:target:container": true,
        "Inside:second_object:container": false,
        "AtBuffer:target:buffer": false,
        "AtBuffer:second_object:buffer": false,
        "OnTable:target": true,
        "OnTable:second_object": true
      },
      "verifier": {
        "p:GripperEmpty": "TRUE",
        "p:Held:target": "FALSE",
        "p:OnTable:target": "TRUE",
        "p:Inside:target:container": "TRUE",
        "p:AtBuffer:target:buffer": "FALSE",
        "p:Held:second_object": "FALSE",
        "p:OnTable:second_object": "TRUE",
        "p:Inside:second_object:container": "FALSE",
        "p:AtBuffer:second_object:buffer": "FALSE",
        "p:Open:container": "TRUE"
      },
      "reasons": {
        "p:GripperEmpty": "cp-disr-d0-verifier-v1:gripper_open_and_no_nearby_blob",
        "p:Held:target": "cp-disr-d0-verifier-v1:open_gripper_or_object_away",
        "p:OnTable:target": "cp-disr-d0-verifier-v1:blob_near_table_plane",
        "p:Inside:target:container": "cp-disr-d0-verifier-v1:xy_inside_container_estimate",
        "p:AtBuffer:target:buffer": "cp-disr-d0-verifier-v1:xy_inside_buffer_estimate",
        "p:Held:second_object": "cp-disr-d0-verifier-v1:open_gripper_or_object_away",
        "p:OnTable:second_object": "cp-disr-d0-verifier-v1:blob_near_table_plane",
        "p:Inside:second_object:container": "cp-disr-d0-verifier-v1:xy_inside_container_estimate",
        "p:AtBuffer:second_object:buffer": "cp-disr-d0-verifier-v1:xy_inside_buffer_estimate",
        "p:Open:container": "cp-disr-d0-verifier-v1:lid_offset_from_container"
      }
    },
    {
      "scenario": "container_closed",
      "skill": null,
      "leaked_hidden_into_verifier": false,
      "qa": {
        "GripperEmpty": true,
        "Held:target": false,
        "Held:second_object": false,
        "Open:container": true,
        "Inside:target:container": true,
        "Inside:second_object:container": false,
        "AtBuffer:target:buffer": false,
        "AtBuffer:second_object:buffer": false,
        "OnTable:target": true,
        "OnTable:second_object": true
      },
      "verifier": {
        "p:GripperEmpty": "TRUE",
        "p:Held:target": "FALSE",
        "p:OnTable:target": "TRUE",
        "p:Inside:target:container": "TRUE",
        "p:AtBuffer:target:buffer": "FALSE",
        "p:Held:second_object": "FALSE",
        "p:OnTable:second_object": "TRUE",
        "p:Inside:second_object:container": "FALSE",
        "p:AtBuffer:second_object:buffer": "FALSE",
        "p:Open:container": "TRUE"
      },
      "reasons": {
        "p:GripperEmpty": "cp-disr-d0-verifier-v1:gripper_open_and_no_nearby_blob",
        "p:Held:target": "cp-disr-d0-verifier-v1:open_gripper_or_object_away",
        "p:OnTable:target": "cp-disr-d0-verifier-v1:blob_near_table_plane",
        "p:Inside:target:container": "cp-disr-d0-verifier-v1:xy_inside_container_estimate",
        "p:AtBuffer:target:buffer": "cp-disr-d0-verifier-v1:xy_inside_buffer_estimate",
        "p:Held:second_object": "cp-disr-d0-verifier-v1:open_gripper_or_object_away",
        "p:OnTable:second_object": "cp-disr-d0-verifier-v1:blob_near_table_plane",
        "p:Inside:second_object:container": "cp-disr-d0-verifier-v1:xy_inside_container_estimate",
        "p:AtBuffer:second_object:buffer": "cp-disr-d0-verifier-v1:xy_inside_buffer_estimate",
        "p:Open:container": "cp-disr-d0-verifier-v1:lid_offset_from_container"
      }
    },
    {
      "scenario": "container_open",
      "skill": "a:OPEN:container:v1",
      "leaked_hidden_into_verifier": false,
      "qa": {
        "GripperEmpty": true,
        "Held:target": false,
        "Held:second_object": false,
        "Open:container": true,
        "Inside:target:container": true,
        "Inside:second_object:container": false,
        "AtBuffer:target:buffer": false,
        "AtBuffer:second_object:buffer": false,
        "OnTable:target": true,
        "OnTable:second_object": true
      },
      "verifier": {
        "p:GripperEmpty": "TRUE",
        "p:Held:target": "FALSE",
        "p:OnTable:target": "TRUE",
        "p:Inside:target:container": "TRUE",
        "p:AtBuffer:target:buffer": "FALSE",
        "p:Held:second_object": "FALSE",
        "p:OnTable:second_object": "TRUE",
        "p:Inside:second_object:container": "FALSE",
        "p:AtBuffer:second_object:buffer": "FALSE",
        "p:Open:container": "TRUE"
      },
      "reasons": {
        "p:GripperEmpty": "cp-disr-d0-verifier-v1:gripper_open_and_no_nearby_blob",
        "p:Held:target": "cp-disr-d0-verifier-v1:open_gripper_or_object_away",
        "p:OnTable:target": "cp-disr-d0-verifier-v1:blob_near_table_plane",
        "p:Inside:target:container": "cp-disr-d0-verifier-v1:xy_inside_container_estimate",
        "p:AtBuffer:target:buffer": "cp-disr-d0-verifier-v1:xy_inside_buffer_estimate",
        "p:Held:second_object": "cp-disr-d0-verifier-v1:open_gripper_or_object_away",
        "p:OnTable:second_object": "cp-disr-d0-verifier-v1:blob_near_table_plane",
        "p:Inside:second_object:container": "cp-disr-d0-verifier-v1:xy_inside_container_estimate",
        "p:AtBuffer:second_object:buffer": "cp-disr-d0-verifier-v1:xy_inside_buffer_estimate",
        "p:Open:container": "cp-disr-d0-verifier-v1:lid_offset_from_container"
      }
    },
    {
      "scenario": "buffer_inside",
      "skill": "a:PLACE_BUFFER:target:buffer:v1",
      "leaked_hidden_into_verifier": false,
      "qa": {
        "GripperEmpty": true,
        "Held:target": false,
        "Held:second_object": false,
        "Open:container": true,
        "Inside:target:container": true,
        "Inside:second_object:container": false,
        "AtBuffer:target:buffer": false,
        "AtBuffer:second_object:buffer": false,
        "OnTable:target": true,
        "OnTable:second_object": true
      },
      "verifier": {
        "p:GripperEmpty": "UNKNOWN",
        "p:Held:target": "UNKNOWN",
        "p:OnTable:target": "TRUE",
        "p:Inside:target:container": "TRUE",
        "p:AtBuffer:target:buffer": "FALSE",
        "p:Held:second_object": "UNKNOWN",
        "p:OnTable:second_object": "TRUE",
        "p:Inside:second_object:container": "FALSE",
        "p:AtBuffer:second_object:buffer": "FALSE",
        "p:Open:container": "TRUE"
      },
      "reasons": {
        "p:GripperEmpty": "cp-disr-d0-verifier-v1:conflicting_or_missing_gripper_evidence",
        "p:Held:target": "cp-disr-d0-verifier-v1:held_evidence_conflict",
        "p:OnTable:target": "cp-disr-d0-verifier-v1:blob_near_table_plane",
        "p:Inside:target:container": "cp-disr-d0-verifier-v1:xy_inside_container_estimate",
        "p:AtBuffer:target:buffer": "cp-disr-d0-verifier-v1:xy_inside_buffer_estimate",
        "p:Held:second_object": "cp-disr-d0-verifier-v1:held_evidence_conflict",
        "p:OnTable:second_object": "cp-disr-d0-verifier-v1:blob_near_table_plane",
        "p:Inside:second_object:container": "cp-disr-d0-verifier-v1:xy_inside_container_estimate",
        "p:AtBuffer:second_object:buffer": "cp-disr-d0-verifier-v1:xy_inside_buffer_estimate",
        "p:Open:container": "cp-disr-d0-verifier-v1:lid_offset_from_container"
      }
    },
    {
      "scenario": "occlusion_unknown",
      "skill": null,
      "leaked_hidden_into_verifier": false,
      "qa": {
        "GripperEmpty": true,
        "Held:target": false,
        "Held:second_object": false,
        "Open:container": true,
        "Inside:target:container": true,
        "Inside:second_object:container": false,
        "AtBuffer:target:buffer": false,
        "AtBuffer:second_object:buffer": false,
        "OnTable:target": true,
        "OnTable:second_object": true
      },
      "verifier": {
        "p:GripperEmpty": "UNKNOWN",
        "p:Held:target": "UNKNOWN",
        "p:OnTable:target": "TRUE",
        "p:Inside:target:container": "TRUE",
        "p:AtBuffer:target:buffer": "FALSE",
        "p:Held:second_object": "UNKNOWN",
        "p:OnTable:second_object": "TRUE",
        "p:Inside:second_object:container": "FALSE",
        "p:AtBuffer:second_object:buffer": "FALSE",
        "p:Open:container": "TRUE"
      },
      "reasons": {
        "p:GripperEmpty": "cp-disr-d0-verifier-v1:conflicting_or_missing_gripper_evidence",
        "p:Held:target": "cp-disr-d0-verifier-v1:held_evidence_conflict",
        "p:OnTable:target": "cp-disr-d0-verifier-v1:blob_near_table_plane",
        "p:Inside:target:container": "cp-disr-d0-verifier-v1:xy_inside_container_estimate",
        "p:AtBuffer:target:buffer": "cp-disr-d0-verifier-v1:xy_inside_buffer_estimate",
        "p:Held:second_object": "cp-disr-d0-verifier-v1:held_evidence_conflict",
        "p:OnTable:second_object": "cp-disr-d0-verifier-v1:blob_near_table_plane",
        "p:Inside:second_object:container": "cp-disr-d0-verifier-v1:xy_inside_container_estimate",
        "p:AtBuffer:second_object:buffer": "cp-disr-d0-verifier-v1:xy_inside_buffer_estimate",
        "p:Open:container": "cp-disr-d0-verifier-v1:lid_offset_from_container"
      }
    }
  ]
}
