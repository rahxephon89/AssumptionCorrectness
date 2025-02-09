(**************************************************************************)
(*                                                                        *)
(*  This file is part of WP plug-in of Frama-C.                           *)
(*                                                                        *)
(*  Copyright (C) 2007-2019                                               *)
(*    CEA (Commissariat a l'energie atomique et aux energies              *)
(*         alternatives)                                                  *)
(*                                                                        *)
(*  you can redistribute it and/or modify it under the terms of the GNU   *)
(*  Lesser General Public License as published by the Free Software       *)
(*  Foundation, version 2.1.                                              *)
(*                                                                        *)
(*  It is distributed in the hope that it will be useful,                 *)
(*  but WITHOUT ANY WARRANTY; without even the implied warranty of        *)
(*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *)
(*  GNU Lesser General Public License for more details.                   *)
(*                                                                        *)
(*  See the GNU Lesser General Public License version 2.1                 *)
(*  for more details (enclosed in the file licenses/LGPLv2.1).            *)
(*                                                                        *)
(**************************************************************************)

open Factory
open Printf

let dkey_main = Wp_parameters.register_category "main"
let dkey_raised = Wp_parameters.register_category "raised"
let dkey_shell = Wp_parameters.register_category "shell"

(* --------- Command Line ------------------- *)

let cmdline () : setup =
  begin
    match Wp_parameters.Model.get () with
    | ["Runtime"] ->
        Wp_parameters.abort
          "Model 'Runtime' is no more available.@\nIt will be reintroduced \
           in a future release."
    | ["Logic"] ->
        Wp_parameters.warning ~once:true
          "Deprecated 'Logic' model.@\nUse 'Typed' with option '-wp-ref' \
           instead." ;
        {
          mheap = Factory.Typed MemTyped.Fits ;
          mvar = Factory.Ref ;
          cint = Cint.Natural ;
          cfloat = Cfloat.Real ;
        }
    | ["Store"] ->
        Wp_parameters.warning ~once:true
          "Deprecated 'Store' model.@\nUse 'Typed' instead." ;
        {
          mheap = Factory.Typed MemTyped.Fits ;
          mvar = Factory.Var ;
          cint = Cint.Natural ;
          cfloat = Cfloat.Real ;
        }
    | spec -> Factory.parse spec
  end

let set_model (s:setup) =
  Wp_parameters.Model.set [Factory.ident s]

(* --------- WP Computer -------------------- *)

let computer () =
  if Wp_parameters.Model.get () = ["Dump"]
  then CfgDump.create ()
  else CfgWP.computer (cmdline ()) (Driver.load_driver ())

(* ------------------------------------------------------------------------ *)
(* --- Memory Model Hypotheses                                          --- *)
(* ------------------------------------------------------------------------ *)

module Models = Set.Make(WpContext.MODEL)
module Fmap = Kernel_function.Map

let wp_iter_model ?ip ?index job =
  begin
    let pool : Models.t Fmap.t ref = ref Fmap.empty in
    Wpo.iter ?ip ?index ~on_goal:(fun wpo ->
        match Wpo.get_index wpo with
        | Wpo.Axiomatic _ -> ()
        | Wpo.Function(kf,_) ->
            let m = Wpo.get_model wpo in
            let ms = try Fmap.find kf !pool with Not_found -> Models.empty in
            if not (Models.mem m ms) then
              pool := Fmap.add kf (Models.add m ms) !pool ;
      ) () ;
    Fmap.iter (fun kf ms -> Models.iter (fun m -> job kf m) ms) !pool
  end

let wp_print_memory_context kf m hyp fmt =
  begin
    let printer = new Printer.extensible_printer () in
    let pp_vdecl = printer#without_annot printer#vdecl in
    Format.fprintf fmt
      "@[<hv 0>@[<hv 3>/*@@@ behavior %s:" (WpContext.MODEL.id m) ;
    List.iter (MemoryContext.pp_clause fmt) hyp ;
    let vkf = Kernel_function.get_vi kf in
    Format.fprintf fmt "@ @]*/@]@\n@[<hov 2>%a;@]@\n"
      pp_vdecl vkf ;
  end

let wp_warn_memory_context () =
  begin
    wp_iter_model
      begin fun kf m ->
        let hyp = WpContext.compute_hypotheses m kf in
        if hyp <> [] then
          Wp_parameters.warning
            ~current:false
            "@[<hv 0>Memory model hypotheses for function '%s':@ %t@]"
            (Kernel_function.get_name kf)
            (wp_print_memory_context kf m hyp)
      end
  end

(* ------------------------------------------------------------------------ *)
(* ---  Printing informations                                           --- *)
(* ------------------------------------------------------------------------ *)

let do_wp_print () =
  (* Printing *)
  if Wp_parameters.Print.get () then
    try
      Wpo.iter ~on_goal:(fun _ -> raise Exit) () ;
      Wp_parameters.result "No proof obligations"
    with Exit ->
      Log.print_on_output
        (fun fmt ->
           Wpo.iter
             ~on_axiomatics:(Wpo.pp_axiomatics fmt)
             ~on_behavior:(Wpo.pp_function fmt)
             ~on_goal:(Wpo.pp_goal_flow fmt) ())

let do_wp_print_for goals =
  if Wp_parameters.Print.get () then
    if Bag.is_empty goals
    then Wp_parameters.result "No proof obligations"
    else Log.print_on_output
        (fun fmt -> Bag.iter (Wpo.pp_goal_flow fmt) goals)

let do_wp_report () =
  begin
    let reports = Wp_parameters.Report.get () in
    let jreport = Wp_parameters.ReportJson.get () in
    if reports <> [] || jreport <> "" then
      begin
        let stats = WpReport.fcstat () in
        begin
          match String.split_on_char ':' jreport with
          | [] | [""] -> ()
          | [joutput] ->
              WpReport.export_json stats ~joutput () ;
          | [jinput;joutput] ->
              WpReport.export_json stats ~jinput ~joutput () ;
          | _ ->
              Wp_parameters.error "Invalid format for option -wp-report-json"
        end ;
        List.iter (WpReport.export stats) reports ;
      end ;
    if Wp_parameters.MemoryContext.get () then
      wp_warn_memory_context ()
  end

(* ------------------------------------------------------------------------ *)
(* ---  Wp Results                                                      --- *)
(* ------------------------------------------------------------------------ *)

let pp_warnings fmt wpo =
  let ws = Wpo.warnings wpo in
  if ws <> [] then
    let n = List.length ws in
    let s = List.exists (fun w -> w.Warning.severe) ws in
    begin
      match s , n with
      | true , 1 -> Format.fprintf fmt " (Degenerated)"
      | true , _ -> Format.fprintf fmt " (Degenerated, %d warnings)" n
      | false , 1 -> Format.fprintf fmt " (Stronger)"
      | false , _ -> Format.fprintf fmt " (Stronger, %d warnings)" n
    end

let launch task =
  let server = ProverTask.server () in
  (** Do on_server_stop save why3 session *)
  Task.spawn server (Task.thread task) ;
  Task.launch server

(* ------------------------------------------------------------------------ *)
(* ---  Prover Stats                                                    --- *)
(* ------------------------------------------------------------------------ *)

let do_wpo_display goal =
  let result = if Wpo.is_trivial goal then "trivial" else "not tried" in
  Wp_parameters.feedback "Goal %s : %s" (Wpo.get_gid goal) result

module PM =
  FCMap.Make(struct
    type t = VCS.prover
    let compare = VCS.cmp_prover
  end)

type pstat = {
  mutable proved : int ;
  mutable unknown : int ;
  mutable interrupted : int ;
  mutable incache : int ;
  mutable failed : int ;
  mutable n_time : int ;   (* nbr of measured times *)
  mutable a_time : float ; (* sum of measured times *)
  mutable u_time : float ; (* max time *)
  mutable d_time : float ; (* min time *)
  mutable steps : int ;
}

module GOALS = Wpo.S.Set

let scheduled = ref 0
let exercised = ref 0
let spy = ref false
let session = ref GOALS.empty
let proved = ref GOALS.empty
let provers = ref PM.empty

let begin_session () = session := GOALS.empty ; spy := true
let clear_session () = session := GOALS.empty
let end_session   () = session := GOALS.empty ; spy := false
let iter_session f  = GOALS.iter f !session

let clear_scheduled () =
  begin
    scheduled := 0 ;
    exercised := 0 ;
    proved := GOALS.empty ;
    provers := PM.empty ;
  end

let get_pstat p =
  try PM.find p !provers with Not_found ->
    let s = {
      proved = 0 ;
      unknown = 0 ;
      interrupted = 0 ;
      failed = 0 ;
      steps = 0 ;
      incache = 0 ;
      n_time = 0 ;
      a_time = 0.0 ;
      u_time = 0.0 ;
      d_time = max_float ;
    } in provers := PM.add p s !provers ; s

let add_step s n =
  if n > s.steps then s.steps <- n

let add_time s t =
  if t > 0.0 then
    begin
      s.n_time <- succ s.n_time ;
      s.a_time <- t +. s.a_time ;
      if t < s.d_time then s.d_time <- t ;
      if t > s.u_time then s.u_time <- t ;
    end

let do_list_scheduled iter_on_goals =
  if not (Wp_parameters.has_dkey VCS.dkey_no_goals_info) then
    begin
      clear_scheduled () ;
      iter_on_goals
        (fun goal ->
           begin
             incr scheduled ;
             if !spy then session := GOALS.add goal !session ;
           end) ;
      let n = !scheduled in
      if n > 1
      then Wp_parameters.feedback "%d goals scheduled" n
      else Wp_parameters.feedback "%d goal scheduled" n ;
    end

let dkey_prover = Wp_parameters.register_category "prover"

let do_wpo_start goal =
  begin
    incr exercised ;
    if Wp_parameters.has_dkey dkey_prover then
      Wp_parameters.feedback "[Qed] Goal %s preprocessing" (Wpo.get_gid goal) ;
  end

let do_wpo_wait () =
  Wp_parameters.feedback ~ontty:`Transient "[wp] Waiting provers..."

let do_progress goal msg =
  begin
    if !scheduled > 0 then
      let pp = int_of_float (100.0 *. float !exercised /. float !scheduled) in
      let pp = max 0 (min 100 pp) in
      Wp_parameters.feedback ~ontty:`Transient "[%02d%%] %s (%s)"
        pp goal.Wpo.po_sid msg ;
  end

(* ------------------------------------------------------------------------ *)
(* ---  Caching                                                         --- *)
(* ------------------------------------------------------------------------ *)

let do_report_cache_usage mode =
  if not (Wp_parameters.has_dkey dkey_shell) &&
     not (Wp_parameters.has_dkey VCS.dkey_no_cache_info)
  then
    let hits = ProverWhy3.get_hits () in
    let miss = ProverWhy3.get_miss () in
    if hits <= 0 && miss <= 0 then
      Wp_parameters.result "[Cache] not used"
    else
      Wp_parameters.result "[Cache]%t"
        begin fun fmt ->
          let sep = ref " " in
          let pp_cache fmt n job =
            if n > 0 then
              ( Format.fprintf fmt "%s%s:%d" !sep job n ; sep := ", " ) in
          match mode with
          | ProverWhy3.NoCache -> ()
          | ProverWhy3.Replay ->
              pp_cache fmt hits "found" ;
              pp_cache fmt miss "missed" ;
              Format.pp_print_newline fmt () ;
          | ProverWhy3.Offline ->
              pp_cache fmt hits "found" ;
              pp_cache fmt miss "failed" ;
              Format.pp_print_newline fmt () ;
          | ProverWhy3.Update | ProverWhy3.Cleanup ->
              pp_cache fmt hits "found" ;
              pp_cache fmt miss "updated" ;
              Format.pp_print_newline fmt () ;
          | ProverWhy3.Rebuild ->
              pp_cache fmt hits "replaced" ;
              pp_cache fmt miss "updated" ;
              Format.pp_print_newline fmt () ;
        end

(* -------------------------------------------------------------------------- *)
(* --- Prover Results                                                     --- *)
(* -------------------------------------------------------------------------- *)

let do_wpo_stat goal prover res =
  let s = get_pstat prover in
  let open VCS in
  if res.cached then s.incache <- succ s.incache ;
  match res.verdict with
  | Checked | NoResult | Computing _ | Invalid | Unknown ->
      s.unknown <- succ s.unknown
  | Stepout | Timeout ->
      s.interrupted <- succ s.interrupted
  | Failed ->
      s.failed <- succ s.failed
  | Valid ->
      if not (Wpo.is_tactic goal) then
        proved := GOALS.add goal !proved ;
      s.proved <- succ s.proved ;
      add_step s res.prover_steps ;
      add_time s res.prover_time ;
      if prover <> Qed then
        add_time (get_pstat Qed) res.solver_time

let do_wpo_result goal prover res =
  if VCS.is_verdict res then
    begin
      if Wp_parameters.Check.get () then
        begin
          let open VCS in
          let ontty = if res.verdict = Checked then `Feedback else `Message in
          Wp_parameters.feedback ~ontty
            "[%a] Goal %s : %a"
            VCS.pp_prover prover (Wpo.get_gid goal)
            VCS.pp_result res ;
        end ;
      if prover = VCS.Qed then do_progress goal "Qed" ;
      do_wpo_stat goal prover res ;
    end

let do_wpo_success goal s =
  if not (Wp_parameters.Check.get ()) then
    if Wp_parameters.Generate.get () then
      match s with
      | None -> ()
      | Some prover ->
          Wp_parameters.feedback ~ontty:`Silent
            "[%a] Goal %s : Valid" VCS.pp_prover prover (Wpo.get_gid goal)
    else
      match s with
      | None ->
          begin
            match Wpo.get_results goal with
            | [p,r] ->
                Wp_parameters.result "[%a] Goal %s : %a%a"
                  VCS.pp_prover p (Wpo.get_gid goal)
                  VCS.pp_result r pp_warnings goal
            | pres ->
                Wp_parameters.result "[Failed] Goal %s%t" (Wpo.get_gid goal)
                  begin fun fmt ->
                    pp_warnings fmt goal ;
                    List.iter
                      (fun (p,r) ->
                         Format.fprintf fmt "@\n%8s: @[<hv>%a@]"
                           (VCS.title_of_prover p) VCS.pp_result r
                      ) pres ;
                  end
          end
      | Some (VCS.Tactical as p) ->
          Wp_parameters.feedback ~ontty:`Silent
            "[%a] Goal %s : Valid"
            VCS.pp_prover p (Wpo.get_gid goal)
      | Some p ->
          let r = Wpo.get_result goal p in
          Wp_parameters.feedback ~ontty:`Silent
            "[%a] Goal %s : %a"
            VCS.pp_prover p (Wpo.get_gid goal)
            VCS.pp_result r

let do_report_time fmt s =
  begin
    if s.n_time > 0 &&
       s.u_time > Rformat.epsilon &&
       not (Wp_parameters.has_dkey VCS.dkey_no_time_info) &&
       not (Wp_parameters.has_dkey VCS.dkey_success_only)
    then
      let mean = s.a_time /. float s.n_time in
      let epsilon = 0.05 *. mean in
      let delta = s.u_time -. s.d_time in
      if delta < epsilon then
        Format.fprintf fmt " (%a)" Rformat.pp_time mean
      else
        let middle = (s.u_time +. s.d_time) *. 0.5 in
        if abs_float (middle -. mean) < epsilon then
          Format.fprintf fmt " (%a-%a)"
            Rformat.pp_time s.d_time
            Rformat.pp_time s.u_time
        else
          Format.fprintf fmt " (%a-%a-%a)"
            Rformat.pp_time s.d_time
            Rformat.pp_time mean
            Rformat.pp_time s.u_time
  end

let do_report_steps fmt s =
  begin
    if s.steps > 0 &&
       not (Wp_parameters.has_dkey VCS.dkey_no_step_info) &&
       not (Wp_parameters.has_dkey VCS.dkey_success_only)
    then
      Format.fprintf fmt " (%d)" s.steps ;
  end

let do_report_stopped fmt s =
  if Wp_parameters.has_dkey VCS.dkey_success_only then
    begin
      let n = s.interrupted + s.unknown in
      if n > 0 then
        Format.fprintf fmt " (unsuccess: %d)" n ;
    end
  else
    begin
      if s.interrupted > 0 then
        Format.fprintf fmt " (interrupted: %d)" s.interrupted ;
      if s.unknown > 0 then
        Format.fprintf fmt " (unknown: %d)" s.unknown ;
      if s.incache > 0 then
        Format.fprintf fmt " (cached: %d)" s.incache ;
    end

let do_report_prover_stats pp_prover fmt (p,s) =
  begin
    let name = VCS.title_of_prover p in
    Format.fprintf fmt "%a %4d " pp_prover name s.proved ;
    do_report_time fmt s ;
    do_report_steps fmt s ;
    do_report_stopped fmt s ;
    if s.failed > 0 then
      Format.fprintf fmt " (failed: %d)" s.failed ;
    Format.fprintf fmt "@\n" ;
  end

let do_report_scheduled () =
  if not (Wp_parameters.has_dkey VCS.dkey_no_goals_info) then
    if Wp_parameters.Generate.get () then
      let plural = if !exercised > 1 then "s" else "" in
      Wp_parameters.result "%d goal%s generated" !exercised plural
    else
      let proved = GOALS.cardinal !proved in
      let mode = ProverWhy3.get_mode () in
      if mode <> ProverWhy3.NoCache then do_report_cache_usage mode ;
      Wp_parameters.result "%t"
        begin fun fmt ->
          Format.fprintf fmt "Proved goals: %4d / %d@\n" proved !scheduled ;
          Pretty_utils.pp_items
            ~min:12 ~align:`Left
            ~title:(fun (prover,_) -> VCS.title_of_prover prover)
            ~iter:(fun f -> PM.iter (fun p s -> f (p,s)) !provers)
            ~pp_title:(fun fmt a -> Format.fprintf fmt "%s:" a)
            ~pp_item:do_report_prover_stats fmt ;
        end

let do_list_scheduled_result () =
  begin
    do_report_scheduled () ;
    clear_scheduled () ;
  end

(* ------------------------------------------------------------------------ *)
(* ---  Proving                                                         --- *)
(* ------------------------------------------------------------------------ *)

type mode = {
  mutable tactical : bool ;
  mutable update : bool ;
  mutable depth : int ;
  mutable width : int ;
  mutable backtrack : int ;
  mutable auto : Strategy.heuristic list ;
  mutable provers : (VCS.mode * VCS.prover) list ;
}

let spawn_wp_proofs_iter ~mode iter_on_goals =
  if mode.tactical || mode.provers<>[] then
    begin
      let server = ProverTask.server () in
      ignore (Wp_parameters.Share.dir ()); (* To prevent further errors *)
      iter_on_goals
        (fun goal ->
           if  mode.tactical
            && not (Wpo.is_trivial goal)
            && (mode.auto <> [] || ProofSession.exists goal)
           then
             ProverScript.spawn
               ~failed:false
               ~auto:mode.auto
               ~depth:mode.depth
               ~width:mode.width
               ~backtrack:mode.backtrack
               ~provers:(List.map snd mode.provers)
               ~start:do_wpo_start
               ~progress:do_progress
               ~result:do_wpo_result
               ~success:do_wpo_success
               goal
           else
             let () = match goal.po_formula with
                      | GoalAnnot vc -> Wp_parameters.debug ~dkey:dkey_main "spawn:%a." Lang.F.pp_pred (snd (vc.goal.sequent))
                      | _ -> Printf.printf "prover other"
             in  
             Prover.spawn goal
               ~delayed:false
               ~start:do_wpo_start
               ~progress:do_progress
               ~result:do_wpo_result
               ~success:do_wpo_success
               mode.provers
        ) ;
      Task.on_server_wait server do_wpo_wait ;
      Task.launch server
    end

let get_prover_names () =
  match Wp_parameters.Provers.get () with [] -> [ "alt-ergo" ] | pnames -> pnames

let compute_provers ~mode =
  mode.provers <- List.fold_right
      (fun pname prvs ->
         match VCS.prover_of_name pname with
         | None -> prvs
         | Some VCS.Tactical ->
             mode.tactical <- true ;
             if pname = "tip" then mode.update <- true ;
             prvs
         | Some prover ->
             (VCS.mode_of_prover_name pname , prover) :: prvs)
      (get_prover_names ()) []

let dump_strategies =
  let once = ref true in
  fun () ->
    if !once then
      ( once := false ;
        Wp_parameters.result "Registered strategies for -wp-auto:%t"
          (fun fmt ->
             Strategy.iter (fun h ->
                 Format.fprintf fmt "@\n  '%s': %s" h#id h#title
               )))

let default_mode () = {
  tactical = false ; update=false ; provers = [] ;
  depth=0 ; width = 0 ; auto=[] ; backtrack = 0 ;
}

let compute_auto ~mode =
  mode.auto <- [] ;
  mode.width <- Wp_parameters.AutoWidth.get () ;
  mode.depth <- Wp_parameters.AutoDepth.get () ;
  mode.backtrack <- max 0 (Wp_parameters.BackTrack.get ()) ;
  let auto = Wp_parameters.Auto.get () in
  if mode.depth <= 0 || mode.width <= 0 then
    ( if auto <> [] then
        Wp_parameters.feedback
          "Auto-search deactivated because of 0-depth or 0-width" )
  else
    begin
      List.iter
        (fun id ->
           if id = "?" then dump_strategies ()
           else
             try mode.auto <- Strategy.lookup ~id :: mode.auto
             with Not_found ->
               Wp_parameters.error ~current:false
                 "Strategy -wp-auto '%s' unknown (ignored)." id
        ) auto ;
      mode.auto <- List.rev mode.auto ;
      if mode.auto <> [] then mode.tactical <- true ;
    end

let do_update_session mode iter =
  if mode.update then
    begin
      let removed = ref 0 in
      let updated = ref 0 in
      let invalid = ref 0 in
      iter
        begin fun goal ->
          let results = Wpo.get_results goal in
          let autoproof (p,r) =
            (p=VCS.Qed) || (VCS.is_auto p && VCS.is_valid r && VCS.autofit r) in
          if List.exists autoproof results then
            begin
              if ProofSession.exists goal then
                (incr removed ; ProofSession.remove goal)
            end
          else
            let scripts = ProofEngine.script (ProofEngine.proof ~main:goal) in
            if scripts <> [] then
              begin
                let keep = function
                  | ProofScript.Prover(p,r) -> VCS.is_auto p && VCS.is_valid r
                  | ProofScript.Tactic(n,_,_) -> n=0
                  | ProofScript.Error _ -> false in
                let strategy = List.filter keep scripts in
                if strategy <> [] then
                  begin
                    incr updated ;
                    ProofSession.save goal (ProofScript.encode strategy)
                  end
                else
                if not (ProofSession.exists goal) then
                  begin
                    incr invalid ;
                    ProofSession.save goal (ProofScript.encode scripts)
                  end
              end
        end ;
      let r = !removed in
      let u = !updated in
      let f = !invalid in
      ( if r = 0 && u = 0 && f = 0 then
          Wp_parameters.result "No updated script." ) ;
      ( if r > 0 then
          let s = if r > 1 then "s" else "" in
          Wp_parameters.result "Updated session with %d new automated proof%s." r s );
      ( if u > 0 then
          let s = if u > 1 then "s" else "" in
          Wp_parameters.result "Updated session with %d new valid script%s." u s ) ;
      ( if f > 0 then
          let s = if f > 1 then "s" else "" in
          Wp_parameters.result "Updated session with %d new script%s to complete." f s );
    end

let do_wp_proofs_iter ?provers ?tip iter =
  let mode = default_mode () in
  Wp_parameters.debug ~dkey:dkey_main "do_wp_proofs_iter @.";
  compute_provers ~mode ;
  compute_auto ~mode ;
  begin match provers with None -> () | Some prvs ->
    mode.provers <- List.map (fun dp -> VCS.BatchMode , VCS.Why3 dp) prvs
  end ;
  begin match tip with None -> () | Some tip ->
    mode.tactical <- tip ;
    mode.update <- tip ;
  end ;
  let spawned = mode.tactical || mode.provers <> [] in
  begin
    if spawned then do_list_scheduled iter ;
    spawn_wp_proofs_iter ~mode iter ;
    if spawned then
      begin
        do_list_scheduled_result () ;
        do_update_session mode iter ;
      end
    else if not (Wp_parameters.Print.get ()) then
      iter do_wpo_display
  end

let do_wp_proofs () = do_wp_proofs_iter (fun f -> Wpo.iter ~on_goal:f ())

let do_wp_proofs_for goals = do_wp_proofs_iter (fun f -> Bag.iter f goals)

(* registered at frama-c (normal) exit *)
let do_cache_cleanup () =
  begin
    let mode = ProverWhy3.get_mode () in
    ProverWhy3.cleanup_cache ~mode ;
    let removed = ProverWhy3.get_removed () in
    if removed > 0 &&
       not (Wp_parameters.has_dkey dkey_shell) &&
       not (Wp_parameters.has_dkey VCS.dkey_no_cache_info)
    then
      Wp_parameters.result "[Cache] removed:%d" removed
  end

(* ------------------------------------------------------------------------ *)
(* ---  Secondary Entry Points                                          --- *)
(* ------------------------------------------------------------------------ *)

(* Deprecated entry point in Dynamic. *)

let deprecated_wp_compute kf bhv ipopt =
  let model = computer () in
  let goals =
    match ipopt with
    | None -> Generator.compute_kf model ?kf ~bhv ()
    | Some ip -> Generator.compute_ip model ip
  in do_wp_proofs_for goals

let deprecated_wp_compute_kf kf bhv prop =
  let model = computer () in
  do_wp_proofs_for (Generator.compute_kf model ?kf ~bhv ~prop ())


let deprecated_wp_compute_ip ip =
  Wp_parameters.warning ~once:true "Dynamic 'wp_compute_ip' is now deprecated." ;
  let model = computer () in
  do_wp_proofs_for (Generator.compute_ip model ip)

let deprecated_wp_compute_call stmt =
  Wp_parameters.warning ~once:true "Dynamic 'wp_compute_ip' is now deprecated." ;
  do_wp_proofs_for (Generator.compute_call (computer ()) stmt)

let deprecated_wp_clear () =
  Wp_parameters.warning ~once:true "Dynamic 'wp_compute_ip' is now deprecated." ;
  Wpo.clear ()

(* ------------------------------------------------------------------------ *)
(* ---  Command-line Entry Points                                       --- *)
(* ------------------------------------------------------------------------ *)

let dkey_logicusage = Wp_parameters.register_category "logicusage"
let dkey_refusage = Wp_parameters.register_category "refusage"
let dkey_builtins = Wp_parameters.register_category "builtins"

let cmdline_run () =
  let wp_main fct =
    Wp_parameters.feedback ~ontty:`Feedback "2Running WP plugin...";
    Ast.compute ();
    Dyncall.compute ();
    if Wp_parameters.has_dkey dkey_logicusage then
      begin
        Printf.printf "logicUsage compute";
        Wp_parameters.feedback ~ontty:`Feedback "logicUsage compute...";
        LogicUsage.compute ();
        LogicUsage.dump ();
      end ;
    if Wp_parameters.has_dkey dkey_refusage then
      begin
        Printf.printf "refusage compute";
        Wp_parameters.feedback ~ontty:`Feedback "Refusage compute...";
        RefUsage.compute ();
        RefUsage.dump ();
      end ;
    let bhv = Wp_parameters.Behaviors.get () in
    let prop = Wp_parameters.Properties.get () in
    (** TODO entry point *)
    let computer = computer () in
    if Wp_parameters.has_dkey dkey_builtins then
      begin
        Printf.printf "built compute";
        Wp_parameters.feedback ~ontty:`Feedback "builtin compute...";
        WpContext.on_context (computer#model,WpContext.Global)
          LogicBuiltins.dump ();
      end ;
    Generator.compute_selection computer ~fct ~bhv ~prop ()
  in
  let fct = Wp_parameters.get_wp () in
  match fct with
  | Wp_parameters.Fct_none -> ()
  | Wp_parameters.Fct_all ->
      begin
        ignore (wp_main fct);
        do_wp_proofs ();
        do_wp_print ();
        do_wp_report ();
      end
  | _ ->
      begin
        let goals = wp_main fct in
        do_wp_proofs_for goals ;
        do_wp_print_for goals ;
        do_wp_report () ;
      end

(* ------------------------------------------------------------------------ *)
(* ---  Register external functions                                     --- *)
(* ------------------------------------------------------------------------ *)

let deprecated name =
  Wp_parameters.warning ~once:true ~current:false
    "Dynamic '%s' now is deprecated. Use `Wp.VC` api instead." name

let register name ty code =
  let _ignore =
    Dynamic.register ~plugin:"Wp" name ty
      ~journalize:false (*LC: Because of Property is not journalizable. *)
      (fun x -> deprecated name ; code x)
  in ()

(* DEPRECATED *)
let () =
  let module OLS = Datatype.List(Datatype.String) in
  let module OKF = Datatype.Option(Kernel_function) in
  let module OP = Datatype.Option(Property) in
  register "wp_compute"
    (Datatype.func3 OKF.ty OLS.ty OP.ty Datatype.unit)
    deprecated_wp_compute

let () =
  let module OKF = Datatype.Option(Kernel_function) in
  let module OLS = Datatype.List(Datatype.String) in
  register "wp_compute_kf"
    (Datatype.func3 OKF.ty OLS.ty OLS.ty Datatype.unit)
    deprecated_wp_compute_kf

let () =
  register "wp_compute_ip"
    (Datatype.func Property.ty Datatype.unit)
    deprecated_wp_compute_ip

let () =
  register "wp_compute_call"
    (Datatype.func Cil_datatype.Stmt.ty Datatype.unit)
    deprecated_wp_compute_call

let () =
  register "wp_clear"
    (Datatype.func Datatype.unit Datatype.unit)
    deprecated_wp_clear

let run = Dynamic.register ~plugin:"Wp" "run"
    (Datatype.func Datatype.unit Datatype.unit)
    ~journalize:true
    cmdline_run

let () =
  let open Datatype in
  begin
    let t_job = func Unit.ty Unit.ty in
    let t_iter = func (func Wpo.S.ty Unit.ty) Unit.ty in
    let register name ty f =
      ignore (Dynamic.register name ty ~plugin:"Wp" ~journalize:false f)
    in
    register "wp_begin_session" t_job  begin_session ;
    register "wp_end_session"   t_job  end_session   ;
    register "wp_clear_session" t_job  clear_session ;
    register "wp_iter_session"  t_iter iter_session  ;
  end

(* ------------------------------------------------------------------------ *)
(* ---  Tracing WP Invocation                                           --- *)
(* ------------------------------------------------------------------------ *)

let pp_wp_parameters fmt =
  begin
    Format.pp_print_string fmt "# frama-c -wp" ;
    if Wp_parameters.RTE.get () then Format.pp_print_string fmt " -wp-rte" ;
    let spec = Wp_parameters.Model.get () in
    if spec <> [] && spec <> ["Typed"] then
      ( let descr = Factory.descr (Factory.parse spec) in
        Format.fprintf fmt " -wp-model '%s'" descr ) ;
    if not (Wp_parameters.Let.get ()) then Format.pp_print_string fmt
        " -wp-no-let" ;
    if Wp_parameters.Let.get () && not (Wp_parameters.Prune.get ())
    then Format.pp_print_string fmt " -wp-no-prune" ;
    if Wp_parameters.Split.get () then Format.pp_print_string fmt " -wp-split" ;
    let tm = Wp_parameters.Timeout.get () in
    if tm <> 10 then Format.fprintf fmt " -wp-timeout %d" tm ;
    let st = Wp_parameters.Steps.get () in
    if st > 0 then Format.fprintf fmt " -wp-steps %d" st ;
    if not (Kernel.SignedOverflow.get ()) then
      Format.pp_print_string fmt " -no-warn-signed-overflow" ;
    if Kernel.UnsignedOverflow.get () then
      Format.pp_print_string fmt " -warn-unsigned-overflow" ;
    if Kernel.SignedDowncast.get () then
      Format.pp_print_string fmt " -warn-signed-downcast" ;
    if Kernel.UnsignedDowncast.get () then
      Format.pp_print_string fmt " -warn-unsigned-downcast" ;
    if not (Wp_parameters.Volatile.get ()) then
      Format.pp_print_string fmt " -wp-no-volatile" ;
    Format.pp_print_string fmt " [...]" ;
    Format.pp_print_newline fmt () ;
  end

let () = Cmdline.run_after_setting_files
    (fun _ ->
       if Wp_parameters.has_dkey dkey_shell then
         Log.print_on_output pp_wp_parameters)

(* -------------------------------------------------------------------------- *)
(* --- Prover Configuration & Detection                                   --- *)
(* -------------------------------------------------------------------------- *)

let () = Cmdline.run_after_configuring_stage Why3Provers.configure

let do_prover_detect () =
  if not !Config.is_gui && Wp_parameters.Detect.get () then
    let provers = Why3Provers.provers () in
    if provers = [] then
      Wp_parameters.result "No Why3 provers detected."
    else
      let open Why3.Whyconf in
      let shortcuts = get_prover_shortcuts (Why3Provers.config ()) in
      let print_prover_shortcuts_for fmt p =
        Why3.Wstdlib.Mstr.iter
          (fun name p' -> if Prover.equal p p' then
              Format.fprintf fmt "%s|" name)
          shortcuts in
      List.iter
        (fun p ->
           Wp_parameters.result "Prover %10s %-6s [%a%a]"
             p.prover_name p.prover_version
             print_prover_shortcuts_for p
             print_prover_parseable_format p
        ) provers

(* ------------------------------------------------------------------------ *)
(* ---  Main Entry Points                                               --- *)
(* ------------------------------------------------------------------------ *)

let rec try_sequence jobs () = match jobs with
  | [] -> ()
  | head :: tail ->
      Extlib.try_finally ~finally:(try_sequence tail) head ()

let sequence jobs () =
  if Wp_parameters.has_dkey dkey_raised
  then List.iter (fun f -> f ()) jobs
  else try_sequence jobs ()

let tracelog () =
  let active_keys = Wp_parameters.get_debug_keys () in
  if active_keys <> [] then begin
    let pp_sep fmt () = Format.pp_print_string fmt "," in
    Wp_parameters.(
      debug "Logging keys: %a."
        (Format.pp_print_list ~pp_sep pp_category) active_keys)
  end

let print_l lst = List.iter (Wp_parameters.debug ~dkey:dkey_main "...%s.") (lst)

let function_name () = 
  let name =  Wp_parameters.FunctionName.get () in
     ignore (Ast.get);
      print_l name
     (* List.iter (printf "%s ") name*)
      


let rec print_list = function 
[] -> ()
| e::l -> (Wp_parameters.debug ~dkey:dkey_main "...%s.") e ; print_list l

let main = sequence [
    (fun () -> Wp_parameters.debug ~dkey:dkey_main "Start WP plugin...@.") ;
    do_prover_detect ;
    cmdline_run ;
    tracelog ;
    (*function_name;*)
    Wp_parameters.reset ;
    (fun () -> Wp_parameters.debug ~dkey:dkey_main "Stop WP plugin...@.") ;
    (*(fun () -> print_l (Wp_parameters.FunctionName.get ())) ;*)
    (*function_name;*)
    
  ]

let () = Cmdline.at_normal_exit do_cache_cleanup
let () = Db.Main.extend main

(* ------------------------------------------------------------------------ *)
