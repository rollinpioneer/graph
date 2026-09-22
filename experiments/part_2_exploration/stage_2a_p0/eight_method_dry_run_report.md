# Eight method dry-run

[
  {
    "task_id": "T_A",
    "method": "B0",
    "case_id": "T_A_dev_00",
    "prior_edge_count": 0,
    "mask_true": 3,
    "candidate_ids": [
      "a:OPEN:container:v1",
      "a:PICK:second_object:v1",
      "a:PICK:target:v1",
      "a:PLACE:second_object:container:v1",
      "a:PLACE:target:container:v1",
      "a:PLACE_BUFFER:second_object:buffer:v1",
      "a:PLACE_BUFFER:target:buffer:v1"
    ],
    "mask": [
      true,
      true,
      true,
      false,
      false,
      false,
      false
    ],
    "selected_candidate_id": "a:OPEN:container:v1",
    "source_prior_edge_count": 0,
    "effective_prior_edge_count": 0,
    "transition": {
      "duration": 7.5999999999995405,
      "reward": 0.0,
      "terminated": false,
      "truncated": false,
      "reason": "CONTINUE",
      "selected_candidate_id": "a:OPEN:container:v1"
    },
    "result": {
      "success": false,
      "terminated": false,
      "truncated": false,
      "reason": "CONTINUE"
    },
    "controller_exit": "NORMAL_TERMINATION",
    "diagnostics": {
      "dp_abs_max": 0.0,
      "dk_abs_max": 0.0,
      "delta_abs_max": 0.0,
      "up_abs_max": 0.0,
      "b0_no_differences": true,
      "b1_no_nominal_successor": false
    },
    "optimizer_steps": 0,
    "loss_computed": false
  },
  {
    "task_id": "T_A",
    "method": "B1",
    "case_id": "T_A_dev_00",
    "prior_edge_count": 0,
    "mask_true": 3,
    "candidate_ids": [
      "a:OPEN:container:v1",
      "a:PICK:second_object:v1",
      "a:PICK:target:v1",
      "a:PLACE:second_object:container:v1",
      "a:PLACE:target:container:v1",
      "a:PLACE_BUFFER:second_object:buffer:v1",
      "a:PLACE_BUFFER:target:buffer:v1"
    ],
    "mask": [
      true,
      true,
      true,
      false,
      false,
      false,
      false
    ],
    "selected_candidate_id": "a:PICK:second_object:v1",
    "source_prior_edge_count": 0,
    "effective_prior_edge_count": 0,
    "transition": {
      "duration": 4.399999999999626,
      "reward": 0.0,
      "terminated": false,
      "truncated": false,
      "reason": "CONTINUE",
      "selected_candidate_id": "a:PICK:second_object:v1"
    },
    "result": {
      "success": false,
      "terminated": false,
      "truncated": false,
      "reason": "CONTINUE"
    },
    "controller_exit": "NORMAL_TERMINATION",
    "diagnostics": {
      "dp_abs_max": 0.0,
      "dk_abs_max": 0.0,
      "delta_abs_max": 0.0,
      "up_abs_max": 0.0,
      "b0_no_differences": false,
      "b1_no_nominal_successor": true
    },
    "optimizer_steps": 0,
    "loss_computed": false
  },
  {
    "task_id": "T_A",
    "method": "B2",
    "case_id": "T_A_dev_00",
    "prior_edge_count": 0,
    "mask_true": 3,
    "candidate_ids": [
      "a:OPEN:container:v1",
      "a:PICK:second_object:v1",
      "a:PICK:target:v1",
      "a:PLACE:second_object:container:v1",
      "a:PLACE:target:container:v1",
      "a:PLACE_BUFFER:second_object:buffer:v1",
      "a:PLACE_BUFFER:target:buffer:v1"
    ],
    "mask": [
      true,
      true,
      true,
      false,
      false,
      false,
      false
    ],
    "selected_candidate_id": "a:OPEN:container:v1",
    "source_prior_edge_count": 0,
    "effective_prior_edge_count": 0,
    "transition": {
      "duration": 7.5999999999995405,
      "reward": 0.0,
      "terminated": false,
      "truncated": false,
      "reason": "CONTINUE",
      "selected_candidate_id": "a:OPEN:container:v1"
    },
    "result": {
      "success": false,
      "terminated": false,
      "truncated": false,
      "reason": "CONTINUE"
    },
    "controller_exit": "NORMAL_TERMINATION",
    "diagnostics": {
      "dp_abs_max": 0.0,
      "dk_abs_max": 0.12592650949954987,
      "delta_abs_max": 0.0,
      "up_abs_max": 0.0,
      "b0_no_differences": false,
      "b1_no_nominal_successor": false
    },
    "optimizer_steps": 0,
    "loss_computed": false
  },
  {
    "task_id": "T_A",
    "method": "Full",
    "case_id": "T_A_dev_00",
    "prior_edge_count": 0,
    "mask_true": 3,
    "candidate_ids": [
      "a:OPEN:container:v1",
      "a:PICK:second_object:v1",
      "a:PICK:target:v1",
      "a:PLACE:second_object:container:v1",
      "a:PLACE:target:container:v1",
      "a:PLACE_BUFFER:second_object:buffer:v1",
      "a:PLACE_BUFFER:target:buffer:v1"
    ],
    "mask": [
      true,
      true,
      true,
      false,
      false,
      false,
      false
    ],
    "selected_candidate_id": "a:OPEN:container:v1",
    "source_prior_edge_count": 0,
    "effective_prior_edge_count": 0,
    "transition": {
      "duration": 7.5999999999995405,
      "reward": 0.0,
      "terminated": false,
      "truncated": false,
      "reason": "CONTINUE",
      "selected_candidate_id": "a:OPEN:container:v1"
    },
    "result": {
      "success": false,
      "terminated": false,
      "truncated": false,
      "reason": "CONTINUE"
    },
    "controller_exit": "NORMAL_TERMINATION",
    "diagnostics": {
      "dp_abs_max": 0.0,
      "dk_abs_max": 0.12570515275001526,
      "delta_abs_max": 0.0,
      "up_abs_max": 0.0,
      "b0_no_differences": false,
      "b1_no_nominal_successor": false
    },
    "optimizer_steps": 0,
    "loss_computed": false
  },
  {
    "task_id": "T_C",
    "method": "B0",
    "case_id": "T_C_dev_00",
    "prior_edge_count": 0,
    "mask_true": 3,
    "candidate_ids": [
      "a:OPEN:container:v1",
      "a:PICK:interferer:v1",
      "a:PICK:target:v1",
      "a:PLACE:interferer:container:v1",
      "a:PLACE:target:container:v1",
      "a:PLACE_BUFFER:interferer:buffer:v1",
      "a:PLACE_BUFFER:target:buffer:v1"
    ],
    "mask": [
      true,
      true,
      true,
      false,
      false,
      false,
      false
    ],
    "selected_candidate_id": "a:OPEN:container:v1",
    "source_prior_edge_count": 0,
    "effective_prior_edge_count": 0,
    "transition": {
      "duration": 7.5999999999995405,
      "reward": 0.0,
      "terminated": false,
      "truncated": false,
      "reason": "CONTINUE",
      "selected_candidate_id": "a:OPEN:container:v1"
    },
    "result": {
      "success": false,
      "terminated": false,
      "truncated": false,
      "reason": "CONTINUE"
    },
    "controller_exit": "NORMAL_TERMINATION",
    "diagnostics": {
      "dp_abs_max": 0.0,
      "dk_abs_max": 0.0,
      "delta_abs_max": 0.0,
      "up_abs_max": 0.0,
      "b0_no_differences": true,
      "b1_no_nominal_successor": false
    },
    "optimizer_steps": 0,
    "loss_computed": false
  },
  {
    "task_id": "T_C",
    "method": "B1",
    "case_id": "T_C_dev_00",
    "prior_edge_count": 0,
    "mask_true": 3,
    "candidate_ids": [
      "a:OPEN:container:v1",
      "a:PICK:interferer:v1",
      "a:PICK:target:v1",
      "a:PLACE:interferer:container:v1",
      "a:PLACE:target:container:v1",
      "a:PLACE_BUFFER:interferer:buffer:v1",
      "a:PLACE_BUFFER:target:buffer:v1"
    ],
    "mask": [
      true,
      true,
      true,
      false,
      false,
      false,
      false
    ],
    "selected_candidate_id": "a:OPEN:container:v1",
    "source_prior_edge_count": 0,
    "effective_prior_edge_count": 0,
    "transition": {
      "duration": 7.5999999999995405,
      "reward": 0.0,
      "terminated": false,
      "truncated": false,
      "reason": "CONTINUE",
      "selected_candidate_id": "a:OPEN:container:v1"
    },
    "result": {
      "success": false,
      "terminated": false,
      "truncated": false,
      "reason": "CONTINUE"
    },
    "controller_exit": "NORMAL_TERMINATION",
    "diagnostics": {
      "dp_abs_max": 0.0,
      "dk_abs_max": 0.0,
      "delta_abs_max": 0.0,
      "up_abs_max": 0.0,
      "b0_no_differences": false,
      "b1_no_nominal_successor": true
    },
    "optimizer_steps": 0,
    "loss_computed": false
  },
  {
    "task_id": "T_C",
    "method": "B2",
    "case_id": "T_C_dev_00",
    "prior_edge_count": 0,
    "mask_true": 3,
    "candidate_ids": [
      "a:OPEN:container:v1",
      "a:PICK:interferer:v1",
      "a:PICK:target:v1",
      "a:PLACE:interferer:container:v1",
      "a:PLACE:target:container:v1",
      "a:PLACE_BUFFER:interferer:buffer:v1",
      "a:PLACE_BUFFER:target:buffer:v1"
    ],
    "mask": [
      true,
      true,
      true,
      false,
      false,
      false,
      false
    ],
    "selected_candidate_id": "a:OPEN:container:v1",
    "source_prior_edge_count": 0,
    "effective_prior_edge_count": 0,
    "transition": {
      "duration": 7.5999999999995405,
      "reward": 0.0,
      "terminated": false,
      "truncated": false,
      "reason": "CONTINUE",
      "selected_candidate_id": "a:OPEN:container:v1"
    },
    "result": {
      "success": false,
      "terminated": false,
      "truncated": false,
      "reason": "CONTINUE"
    },
    "controller_exit": "NORMAL_TERMINATION",
    "diagnostics": {
      "dp_abs_max": 0.0,
      "dk_abs_max": 0.11453467607498169,
      "delta_abs_max": 0.0,
      "up_abs_max": 0.0,
      "b0_no_differences": false,
      "b1_no_nominal_successor": false
    },
    "optimizer_steps": 0,
    "loss_computed": false
  },
  {
    "task_id": "T_C",
    "method": "Full",
    "case_id": "T_C_train_01",
    "prior_edge_count": 1,
    "mask_true": 3,
    "candidate_ids": [
      "a:OPEN:container:v1",
      "a:PICK:interferer:v1",
      "a:PICK:target:v1",
      "a:PLACE:interferer:container:v1",
      "a:PLACE:target:container:v1",
      "a:PLACE_BUFFER:interferer:buffer:v1",
      "a:PLACE_BUFFER:target:buffer:v1"
    ],
    "mask": [
      true,
      true,
      true,
      false,
      false,
      false,
      false
    ],
    "selected_candidate_id": "a:PICK:interferer:v1",
    "source_prior_edge_count": 1,
    "effective_prior_edge_count": 1,
    "transition": {
      "duration": 4.249999999999643,
      "reward": 0.0,
      "terminated": false,
      "truncated": false,
      "reason": "CONTINUE",
      "selected_candidate_id": "a:PICK:interferer:v1"
    },
    "result": {
      "success": false,
      "terminated": false,
      "truncated": false,
      "reason": "CONTINUE"
    },
    "controller_exit": "NORMAL_TERMINATION",
    "diagnostics": {
      "dp_abs_max": 0.0817154124379158,
      "dk_abs_max": 0.14179840683937073,
      "delta_abs_max": 9.654883615439758e-05,
      "up_abs_max": 0.0021780431270599365,
      "b0_no_differences": false,
      "b1_no_nominal_successor": false
    },
    "optimizer_steps": 0,
    "loss_computed": false
  }
]
