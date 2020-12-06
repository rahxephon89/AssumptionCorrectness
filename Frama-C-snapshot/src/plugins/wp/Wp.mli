(** {b WP Public API} *)
module Wp_parameters : sig
# 1 "./wp_parameters.mli"
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

include Plugin.S

val reset : unit -> unit

(** {2 Function Selection} *)

type functions =
  | Fct_none
  | Fct_all
  | Fct_skip of Cil_datatype.Kf.Set.t
  | Fct_list of Cil_datatype.Kf.Set.t

val get_kf : unit -> functions
val get_wp : unit -> functions
val iter_fct : (Kernel_function.t -> unit) -> functions -> unit
val iter_kf : (Kernel_function.t -> unit) -> unit
val iter_wp : (Kernel_function.t -> unit) -> unit

(** {2 Goal Selection} *)

module WP          : Parameter_sig.Bool
module Behaviors   : Parameter_sig.String_list
module Properties  : Parameter_sig.String_list
module StatusAll   : Parameter_sig.Bool
module StatusTrue  : Parameter_sig.Bool
module StatusFalse : Parameter_sig.Bool
module StatusMaybe : Parameter_sig.Bool

(** {2 Model Selection} *)

val has_dkey : category -> bool

module Model : Parameter_sig.String_list
module ByValue : Parameter_sig.String_set
module ByRef : Parameter_sig.String_set
module InHeap : Parameter_sig.String_set
module AliasInit: Parameter_sig.Bool
module InCtxt : Parameter_sig.String_set
module ExternArrays: Parameter_sig.Bool
module Literals : Parameter_sig.Bool
module Volatile : Parameter_sig.Bool

module Region: Parameter_sig.Bool
module Region_rw: Parameter_sig.Bool
module Region_pack: Parameter_sig.Bool
module Region_flat: Parameter_sig.Bool
module Region_annot: Parameter_sig.Bool
module Region_inline: Parameter_sig.Bool
module Region_fixpoint: Parameter_sig.Bool
module Region_cluster: Parameter_sig.Bool

(** {2 Computation Strategies} *)

module Init: Parameter_sig.Bool
module InitWithForall: Parameter_sig.Bool
module BoundForallUnfolding: Parameter_sig.Int
module RTE: Parameter_sig.Bool
module Simpl: Parameter_sig.Bool
module Let: Parameter_sig.Bool
module Core: Parameter_sig.Bool
module Prune: Parameter_sig.Bool
module Clean: Parameter_sig.Bool
module Filter: Parameter_sig.Bool
module Parasite: Parameter_sig.Bool
module Prenex: Parameter_sig.Bool
module Bits: Parameter_sig.Bool
module Ground: Parameter_sig.Bool
module Reduce: Parameter_sig.Bool
module ExtEqual : Parameter_sig.Bool
module UnfoldAssigns : Parameter_sig.Bool
module Split: Parameter_sig.Bool
module SplitDepth: Parameter_sig.Int
module DynCall : Parameter_sig.Bool
module SimplifyIsCint : Parameter_sig.Bool
module SimplifyLandMask : Parameter_sig.Bool
module SimplifyForall : Parameter_sig.Bool
module SimplifyType : Parameter_sig.Bool
module CalleePreCond : Parameter_sig.Bool
module PrecondWeakening : Parameter_sig.Bool

(** {2 Prover Interface} *)

module Detect: Parameter_sig.Bool
module Generate:Parameter_sig.Bool
module Provers: Parameter_sig.String_list
module Cache: Parameter_sig.String
module Drivers: Parameter_sig.String_list
module Script: Parameter_sig.String
module UpdateScript: Parameter_sig.Bool
module Timeout: Parameter_sig.Int
module TimeExtra: Parameter_sig.Int
module TimeMargin: Parameter_sig.Int
module CoqTimeout: Parameter_sig.Int
module CoqCompiler : Parameter_sig.String
module CoqIde : Parameter_sig.String
module CoqProject : Parameter_sig.String
module Steps: Parameter_sig.Int
module Procs: Parameter_sig.Int
module ProofTrace: Parameter_sig.Bool
module CoqLibs: Parameter_sig.String_list
module CoqTactic: Parameter_sig.String
module Hints: Parameter_sig.Int
module TryHints: Parameter_sig.Bool
module Why3Flags: Parameter_sig.String_list
module AltErgo: Parameter_sig.String
module AltGrErgo: Parameter_sig.String
module AltErgoLibs: Parameter_sig.String_list
module AltErgoFlags: Parameter_sig.String_list

module Auto: Parameter_sig.String_list
module AutoDepth: Parameter_sig.Int
module AutoWidth: Parameter_sig.Int
module BackTrack: Parameter_sig.Int

(** {2 Proof Obligations} *)

module TruncPropIdFileName: Parameter_sig.Int
module Print: Parameter_sig.Bool
module Report: Parameter_sig.String_list
module ReportJson: Parameter_sig.String
module ReportName: Parameter_sig.String
module MemoryContext: Parameter_sig.Bool
module Check: Parameter_sig.Bool

(** test plugin *)
module FunctionName: Parameter_sig.String_list

(** {2 Getters} *)

val has_out : unit -> bool
val has_session : unit -> bool
val get_session : unit -> string
val get_session_dir : string -> string
val get_output : unit -> string
val get_output_dir : string -> string
val make_output_dir : string -> unit
val get_overflows : unit -> bool

(** {2 Debugging Categories} *)
val has_print_generated: unit -> bool
val print_generated: ?header:string -> string -> unit
(** print the given file if the debugging category
    "print-generated" is set *)
val cat_print_generated: category
end
module Ctypes : sig
# 1 "./ctypes.mli"
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

(* -------------------------------------------------------------------------- *)
(** C-Types                                                                   *)
(* -------------------------------------------------------------------------- *)

open Cil_types

(** Runtime integers. *)
type c_int =
  | CBool
  | UInt8
  | SInt8
  | UInt16
  | SInt16
  | UInt32
  | SInt32
  | UInt64
  | SInt64

(** Runtime floats. *)
type c_float =
  | Float32
  | Float64

(** Array objects, with both the head view and the flatten view. *)
type arrayflat = {
  arr_size     : int ; (** number of elements in the array *)
  arr_dim      : int ; (** number of dimensions in the array *)
  arr_cell     : typ ; (** type of elementary cells of the flatten array. Never an array. *)
  arr_cell_nbr : int ; (** number of elementary cells in the flatten array *)
}

type arrayinfo = {
  arr_element  : typ ;   (** type of the elements of the array *)
  arr_flat : arrayflat option;
}

(** Type of variable, inits, field or assignable values.
    Abstract view of unrolled C types without attribute. *)
type c_object =
  | C_int of c_int
  | C_float of c_float
  | C_pointer of typ
  | C_comp of compinfo
  | C_array of arrayinfo

val object_of_pointed: c_object -> c_object
val object_of_array_elem : c_object -> c_object
val object_of_logic_type : logic_type -> c_object
val object_of_logic_pointed : logic_type -> c_object

(** {2 Utilities} *)

val i_iter: (c_int -> unit) -> unit
val f_iter: (c_float -> unit) -> unit

val i_memo : (c_int -> 'a) -> c_int -> 'a
(** memoized, not-projectified *)

val f_memo : (c_float -> 'a) -> c_float -> 'a
(** memoized, not-projectified *)

val is_char : c_int -> bool
val c_char : unit -> c_int     (** Returns the type of [char] *)
val c_bool : unit -> c_int     (** Returns the type of [int] *)
val c_ptr  : unit -> c_int     (** Returns the type of pointers *)

val c_int    : ikind -> c_int   (** Conforms to {Cil.theMachine} *)
val c_float  : fkind -> c_float (** Conforms to {Cil.theMachine} *)
val object_of : typ -> c_object

val is_pointer : c_object -> bool

val char : char -> int64
val constant : exp -> int64

val get_int : exp -> int option
val get_int64 : exp -> int64 option

val signed : c_int -> bool  (** [true] if signed *)
val bounds: c_int -> Integer.t * Integer.t (** domain, bounds included *)

val i_bits : c_int -> int (** size in bits *)
val i_bytes : c_int -> int (** size in bytes *)
val f_bits : c_float -> int (** size in bits *)
val f_bytes : c_float -> int (** size in bytes *)
val p_bits : unit -> int (** pointer size in bits *)
val p_bytes : unit -> int (** pointer size in bits *)

val sub_c_int: c_int -> c_int -> bool

val equal_float : c_float -> c_float -> bool

val sizeof_defined : c_object -> bool
val sizeof_object : c_object -> int
val bits_sizeof_comp : compinfo -> int
val bits_sizeof_array : arrayinfo -> int
val bits_sizeof_object : c_object -> int
val field_offset : fieldinfo -> int

val no_infinite_array : c_object -> bool

val is_comp : c_object -> compinfo -> bool
val is_array : c_object -> elt:c_object -> bool
val get_array : c_object -> ( c_object * int option ) option
val get_array_size : c_object -> int option
val get_array_dim : c_object -> int
val array_size : arrayinfo -> int option
val array_dimensions : arrayinfo -> c_object * int option list
(** Returns the list of dimensions the array consists of.
    None-dimension means undefined one. *)
val dimension_of_object : c_object -> (int * int) option
(** Returns None for 1-dimension objects, and Some(d,N) for d-matrix with N cells *)

val i_convert : c_int -> c_int -> c_int
val f_convert : c_float -> c_float -> c_float
val promote : c_object -> c_object -> c_object

val pp_int : Format.formatter -> c_int -> unit
val pp_float : Format.formatter -> c_float -> unit
val pp_object : Format.formatter -> c_object -> unit

val basename : c_object -> string
val compare : c_object -> c_object -> int
val equal : c_object -> c_object -> bool
val hash : c_object -> int
val pretty : Format.formatter -> c_object -> unit

module C_object: Datatype.S with type t = c_object

module AinfoComparable :
sig
  type t = arrayinfo
  val compare : t -> t -> int
  val equal : t -> t -> bool
  val hash : t -> int
end

val compare_c_int : c_int -> c_int -> int
val compare_c_float : c_float -> c_float -> int

val compare_ptr_conflated : c_object -> c_object -> int
(** same as {!compare} but all PTR are considered the same *)
end
module Clabels : sig
# 1 "./clabels.mli"
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

(* -------------------------------------------------------------------------- *)
(** Normalized C-labels                                                       *)
(* -------------------------------------------------------------------------- *)

(**
    Structural representation of logic labels.
    Compatible with pervasives comparison and structural equality.
*)

type c_label

val is_here : c_label -> bool
val mem : c_label -> c_label list -> bool
val equal : c_label -> c_label -> bool

module T : sig type t = c_label val compare : t -> t -> int end
module LabelMap : FCMap.S with type key = c_label
module LabelSet : FCSet.S with type elt = c_label

val pre : c_label
val here : c_label
val next : c_label
val init : c_label
val post : c_label
val break : c_label
val continue : c_label
val default : c_label
val at_exit : c_label
val loopentry : c_label
val loopcurrent : c_label
val old : c_label

val formal : string -> c_label

val case : int64 -> c_label
val stmt : Cil_types.stmt -> c_label
val loop_entry : Cil_types.stmt -> c_label
val loop_current : Cil_types.stmt -> c_label

val to_logic : c_label -> Cil_types.logic_label
val of_logic : Cil_types.logic_label -> c_label
(** Assumes the logic label only comes from normalized or non-ambiguous
    labels. Ambiguous labels are: Old, LoopEntry and LoopCurrent, since
    they points to different program points dependending on the context. *)

val pretty : Format.formatter -> c_label -> unit

open Cil_types

val name : logic_label -> string
val lookup : (logic_label * logic_label) list -> string -> logic_label
(** [lookup bindings lparam] retrieves the actual label
    for the label in [bindings] for label parameter [lparam]. *)
end
module MemoryContext : sig
# 1 "./MemoryContext.mli"
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

open Cil_types

type param = NotUsed | ByAddr | ByValue | ByShift | ByRef | InContext | InArray

val pp_param : Format.formatter -> param -> unit

type partition

val empty : partition
val set : varinfo -> param -> partition -> partition

type zone =
  | Var of varinfo   (** [&x] the cell x *)
  | Ptr of varinfo   (** [p] the cell pointed by p *)
  | Arr of varinfo   (** [p+(..)] the cell and its neighbors pointed by p *)

type clause =
  | Valid of zone
  | Separated of zone list list

(** Build the separation clause from a partition,
    including the clauses related to the pointer validity *)
val requires : partition -> clause list

val pp_zone : Format.formatter -> zone -> unit
val pp_clause : Format.formatter -> clause -> unit
end
module LogicUsage : sig
# 1 "./LogicUsage.mli"
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

(* -------------------------------------------------------------------------- *)
(* --- Dependencies of Logic Definitions                                  --- *)
(* -------------------------------------------------------------------------- *)

open Cil_types
open Cil_datatype
open Clabels

val basename : varinfo -> string (** Trims the original name *)

type logic_lemma = {
  lem_name : string ;
  lem_position : Filepath.position ;
  lem_axiom : bool ;
  lem_types : string list ;
  lem_labels : logic_label list ;
  lem_property : predicate ;
  lem_depends : logic_lemma list ; (** in reverse order *)
}

type axiomatic = {
  ax_name : string ;
  ax_position : Filepath.position ;
  ax_property : Property.t ;
  mutable ax_types : logic_type_info list ;
  mutable ax_logics : logic_info list ;
  mutable ax_lemmas : logic_lemma list ;
  mutable ax_reads : Varinfo.Set.t ; (* read-only *)
}

type logic_section =
  | Toplevel of int
  | Axiomatic of axiomatic

val compute : unit -> unit (** To force computation *)

val ip_lemma : logic_lemma -> Property.t
val iter_lemmas : (logic_lemma -> unit) -> unit
val logic_lemma : string -> logic_lemma
val axiomatic : string -> axiomatic
val section_of_lemma : string -> logic_section
val section_of_type : logic_type_info -> logic_section
val section_of_logic : logic_info -> logic_section
val proof_context : unit -> logic_lemma list
(** Lemmas that are not in an axiomatic. *)

val is_recursive : logic_info -> bool
val get_induction_labels : logic_info -> string -> LabelSet.t LabelMap.t
(** Given an inductive [phi{...A...}].
    Whenever in [case C{...B...}] we have a call to [phi{...B...}],
    then [A] belongs to [(induction phi C).[B]]. *)

val get_name : logic_info -> string
val pp_profile : Format.formatter -> logic_info -> unit

val dump : unit -> unit (** Print on output *)
end
module RefUsage : sig
# 1 "./RefUsage.mli"
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

(* -------------------------------------------------------------------------- *)
(* --- Variable Analysis                                                  --- *)
(* -------------------------------------------------------------------------- *)

open Cil_types

(** By lattice order of usage *)
type access =
  | NoAccess (** Never used *)
  | ByRef   (** Only used as ["*x"],   equals to [load(shift(load(&x),0))] *)
  | ByArray (** Only used as ["x[_]"], equals to [load(shift(load(&x),_))] *)
  | ByValue (** Only used as ["x"],    equals to [load(&x)] *)
  | ByAddr  (** Widely used, potentially up to ["&x"] *)

val get : ?kf:kernel_function -> ?init:bool -> varinfo -> access

val iter: ?kf:kernel_function -> ?init:bool -> (varinfo -> access -> unit) -> unit

val print : varinfo -> access -> Format.formatter -> unit
val dump : unit -> unit
val compute : unit -> unit
end
module NormAtLabels : sig
# 1 "./normAtLabels.mli"
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

open Cil_types
open Clabels

(* exception LabelError of logic_label *)
val catch_label_error : exn -> string -> string -> unit

type label_mapping

val labels_empty : label_mapping
val labels_fct_pre : label_mapping
val labels_fct_post : label_mapping
val labels_fct_assigns : label_mapping
val labels_assert_before : kf:kernel_function -> stmt -> label_mapping
val labels_assert_after : kf:kernel_function -> stmt -> c_label option -> label_mapping
val labels_loop_inv : established:bool -> stmt -> label_mapping
val labels_loop_assigns : stmt -> label_mapping
val labels_stmt_pre : kf:kernel_function -> stmt -> label_mapping
val labels_stmt_post : kf:kernel_function -> stmt -> c_label option -> label_mapping
val labels_stmt_assigns : kf:kernel_function -> stmt -> c_label option -> label_mapping
val labels_predicate : (logic_label * logic_label) list -> label_mapping
val labels_axiom : label_mapping

val preproc_annot : label_mapping -> predicate -> predicate

val preproc_assigns : label_mapping -> from list -> from list
end
module WpPropId : sig
# 1 "./wpPropId.mli"
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

open Cil_types
open LogicUsage

(** Beside the property identification, it can be found in different contexts
 * depending on which part of the computation is involved.
 * For instance, properties on loops are split in 2 parts : establishment and
 * preservation.
*)

(** Property.t information and kind of PO (establishment, preservation, etc) *)
type prop_id

(** returns the annotation which lead to the given PO.
    Dynamically exported.
*)
val property_of_id : prop_id -> Property.t

val source_of_id : prop_id -> Filepath.position

(*----------------------------------------------------------------------------*)

module PropId : Datatype.S with type t = prop_id

(*----------------------------------------------------------------------------*)

val compare_prop_id : prop_id -> prop_id -> int

val tactical : gid:string -> prop_id
val is_check : prop_id -> bool
val is_tactic : prop_id -> bool
val is_assigns : prop_id -> bool
val is_requires : Property.t -> bool
val is_loop_preservation : prop_id -> stmt option

(** test if the prop_id does not have a [no_wp:] in its name(s). *)
val select_default : prop_id -> bool

(** test if the prop_id has to be selected for the asked name.
    Also returns a debug message to explain then answer. Includes
    a test for [no_wp:]. *)
val select_by_name : string list -> prop_id -> bool

(** test if the prop_id has to be selected when we want to select the call
 * precondition the the [stmt] call (None means all the call preconditions).
 * Also returns a debug message to explain then answer. *)
val select_call_pre : stmt -> Property.t option -> prop_id -> bool

(*----------------------------------------------------------------------------*)

val prop_id_keys : prop_id -> string list * string list (* required , hints *)

val get_propid : prop_id -> string (** Unique identifier of [prop_id] *)
val get_legacy : prop_id -> string (** Unique legacy identifier of [prop_id] *)
val pp_propid : Format.formatter -> prop_id -> unit (** Print unique id of [prop_id] *)

type prop_kind =
  | PKTactic      (** tactical sub-goal *)
  | PKCheck       (** internal check *)
  | PKProp        (** normal property *)
  | PKEstablished (** computation related to a loop property before the loop. *)
  | PKPreserved   (** computation related to a loop property inside the loop. *)
  | PKPropLoop    (** loop property used as hypothesis inside a loop. *)
  | PKVarDecr     (** computation related to the decreasing of a variant in a loop *)
  | PKVarPos      (** computation related to a loop variant being positive *)
  | PKAFctOut     (** computation related to the function assigns on normal termination *)
  | PKAFctExit    (** computation related to the function assigns on exit termination *)
  | PKPre of kernel_function * stmt * Property.t (** precondition for function
                                                     at stmt, property of the require. Many information that should come
                                                     from the p_prop part of the prop_id, but in the PKPre case,
                                                     it seems that it is hidden in a IPBlob property ! *)

val pretty : Format.formatter -> prop_id -> unit
val pretty_context : Description.kf -> Format.formatter -> prop_id -> unit
val pretty_local : Format.formatter -> prop_id -> unit

(** Short description of the kind of PO *)
val label_of_prop_id: prop_id -> string

(** TODO: should probably be somewhere else *)
val string_of_termination_kind : termination_kind -> string

val num_of_bhv_from : funbehavior -> from -> int
(*----------------------------------------------------------------------------*)

val mk_code_annot_ids : kernel_function -> stmt -> code_annotation -> prop_id list

val mk_assert_id : kernel_function -> stmt -> code_annotation -> prop_id

(** Invariant establishment and preservation *)
val mk_loop_inv_id : kernel_function -> stmt ->
  established:bool -> code_annotation -> prop_id

(** Invariant used as hypothesis *)
val mk_inv_hyp_id : kernel_function -> stmt -> code_annotation -> prop_id

(** Variant decrease *)
val mk_var_decr_id : kernel_function -> stmt -> code_annotation -> prop_id

(** Variant positive *)
val mk_var_pos_id : kernel_function -> stmt -> code_annotation -> prop_id

(** \from property of loop assigns. Must not be [FromAny] *)
val mk_loop_from_id : kernel_function -> stmt -> code_annotation ->
  from -> prop_id

(** \from property of function or statement behavior assigns.
    Must not be [FromAny] *)
val mk_bhv_from_id :
  kernel_function -> kinstr -> string list -> funbehavior ->
  from -> prop_id

(** \from property of function behavior assigns. Must not be [FromAny]. *)
val mk_fct_from_id : kernel_function -> funbehavior ->
  termination_kind -> from -> prop_id

(** disjoint behaviors property.
    See {!Property.ip_of_disjoint} for more information
*)
val mk_disj_bhv_id :
  kernel_function * kinstr * string list * string list -> prop_id

(** complete behaviors property.
    See {!Property.ip_of_complete} for more information
*)
val mk_compl_bhv_id :
  kernel_function * kinstr * string list * string list -> prop_id

val mk_decrease_id : kernel_function * kinstr * variant -> prop_id

(** axiom identification *)
val mk_lemma_id : logic_lemma -> prop_id

val mk_stmt_assigns_id :
  kernel_function -> stmt -> string list -> funbehavior ->
  from list -> prop_id option

val mk_loop_assigns_id : kernel_function -> stmt -> code_annotation ->
  from list -> prop_id option

(** function assigns *)
val mk_fct_assigns_id : kernel_function -> funbehavior ->
  termination_kind -> from list -> prop_id option

val mk_pre_id : kernel_function -> kinstr -> funbehavior ->
  identified_predicate -> prop_id

val mk_stmt_post_id : kernel_function -> stmt -> funbehavior ->
  termination_kind * identified_predicate -> prop_id

val mk_fct_post_id : kernel_function -> funbehavior ->
  termination_kind * identified_predicate -> prop_id

(** [mk_call_pre_id called_kf s_call called_pre] *)
val mk_call_pre_id : kernel_function -> stmt ->
  Property.t -> Property.t -> prop_id

val mk_property : Property.t -> prop_id

val mk_check : Property.t -> prop_id

(*----------------------------------------------------------------------------*)

type a_kind = LoopAssigns | StmtAssigns
type assigns_desc = private {
  a_label : Clabels.c_label ;
  a_stmt : Cil_types.stmt option ;
  a_kind : a_kind ;
  a_assigns : Cil_types.assigns ;
}
val pp_assigns_desc : Format.formatter -> assigns_desc -> unit

type effect_source = FromCode | FromCall | FromReturn
type assigns_info = prop_id * assigns_desc
val assigns_info_id : assigns_info -> prop_id

type assigns_full_info = private
    AssignsLocations of assigns_info
  | AssignsAny of assigns_desc
  | NoAssignsInfo

val empty_assigns_info : assigns_full_info

val mk_assigns_info : prop_id -> assigns_desc -> assigns_full_info
val mk_stmt_any_assigns_info : stmt -> assigns_full_info
val mk_kf_any_assigns_info : unit -> assigns_full_info
val mk_loop_any_assigns_info : stmt -> assigns_full_info

val pp_assign_info : string -> Format.formatter -> assigns_full_info -> unit
val merge_assign_info :
  assigns_full_info -> assigns_full_info -> assigns_full_info

val mk_loop_assigns_desc : stmt -> from list -> assigns_desc

val mk_stmt_assigns_desc : stmt -> from list -> assigns_desc

val mk_asm_assigns_desc : stmt -> assigns_desc

val mk_kf_assigns_desc : from list -> assigns_desc

val mk_init_assigns : assigns_desc

val is_call_assigns : assigns_desc -> bool

(*----------------------------------------------------------------------------*)

type axiom_info = prop_id * LogicUsage.logic_lemma

val mk_axiom_info : LogicUsage.logic_lemma -> axiom_info
val pp_axiom_info : Format.formatter -> axiom_info -> unit

type pred_info = (prop_id * Cil_types.predicate)

val mk_pred_info : prop_id -> Cil_types.predicate -> pred_info
val pred_info_id : pred_info -> prop_id
val pp_pred_of_pred_info : Format.formatter -> pred_info -> unit
val pp_pred_info : Format.formatter -> pred_info -> unit

(*----------------------------------------------------------------------------*)

(** [mk_part pid (k, n)] build the identification for the [k/n] part of [pid].*)
val mk_part : prop_id -> (int * int) -> prop_id

(** get the 'kind' information. *)
val kind_of_id : prop_id -> prop_kind

(** get the 'part' information. *)
val parts_of_id : prop_id -> (int * int) option

(** How many subproofs *)
val subproofs : prop_id -> int

(** subproof index of this propr_id *)
val subproof_idx : prop_id -> int

val get_induction : prop_id -> stmt option

(*----------------------------------------------------------------------------*)
end
module Mcfg : sig
# 1 "./mcfg.ml"
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

open Cil_types

type scope =
  | SC_Global
  | SC_Function_in    (* Just before the pre-state *)
  | SC_Function_frame (* Just after the introduction of formals *)
  | SC_Function_out   (* Post-state *)
  | SC_Block_in
  | SC_Block_out

module type Export =
sig
  type pred
  type decl
  val export_section : Format.formatter -> string -> unit
  val export_goal : Format.formatter -> string -> pred -> unit
  val export_decl : Format.formatter -> decl -> unit
end

module type Splitter =
sig
  type pred
  val simplify : pred -> pred
  val split : bool -> pred -> pred Bag.t
end

(**
 * This is what is really needed to propagate something through the CFG.
 * Usually, the propagated thing should be a predicate,
 * but it can be more sophisticated like lists of predicates,
 * or maybe a structure to keep hypotheses and goals separated.
 * Moreover, proof obligations may also need to be handled.
 **)
module type S = sig

  type t_env
  type t_prop

  val convert_pred: t_prop list  -> unit
  val convert_pred: t_prop -> t_prop  -> unit
  val pretty : Format.formatter -> t_prop -> unit
  val merge : t_env -> t_prop -> t_prop -> t_prop
  val empty : t_prop
  (*val pp_pred: Format.formatter -> Lang.F.pred -> unit*)

  (** optionally init env with user logic variables *)
  val new_env : ?lvars:Cil_types.logic_var list -> kernel_function -> t_env

  val add_axiom : WpPropId.prop_id -> LogicUsage.logic_lemma -> unit
  val add_hyp  : t_env -> WpPropId.pred_info -> t_prop -> t_prop
  val add_goal : t_env -> WpPropId.pred_info -> t_prop -> t_prop

  val add_assigns : t_env -> WpPropId.assigns_info -> t_prop -> t_prop

  (** [use_assigns env hid kind assgn goal] performs the havoc on the goal.
   * [hid] should be [None] iff [assgn] is [WritesAny],
   * and tied to the corresponding identified_property otherwise.*)
  val use_assigns : t_env -> stmt option -> WpPropId.prop_id option ->
    WpPropId.assigns_desc -> t_prop -> t_prop

  val label  : t_env -> stmt option -> Clabels.c_label -> t_prop -> t_prop
  val init : t_env -> varinfo -> init option -> t_prop -> t_prop
  val const : t_env -> varinfo -> t_prop -> t_prop
  val assign : t_env -> stmt -> lval -> exp -> t_prop -> t_prop
  val return : t_env -> stmt -> exp option -> t_prop -> t_prop
  val test : t_env -> stmt -> exp -> t_prop -> t_prop -> t_prop
  val switch : t_env -> stmt -> exp -> (exp list * t_prop) list -> t_prop -> t_prop

  val has_init : t_env -> bool

  val loop_entry : t_prop -> t_prop
  val loop_step : t_prop -> t_prop

  (* -------------------------------------------------------------------------- *)
  (* --- Call Rules                                                         --- *)
  (* -------------------------------------------------------------------------- *)

  val call_dynamic : t_env -> stmt ->
    WpPropId.prop_id -> exp -> (kernel_function * t_prop) list -> t_prop

  val call_goal_precond : t_env -> stmt ->
    kernel_function -> exp list ->
    pre: WpPropId.pred_info list ->
    t_prop -> t_prop

  val call : t_env -> stmt ->
    lval option -> kernel_function -> exp list ->
    pre:     WpPropId.pred_info list ->
    post:    WpPropId.pred_info list ->
    pexit:   WpPropId.pred_info list ->
    assigns: assigns ->
    p_post: t_prop ->
    p_exit: t_prop ->
    t_prop

  (* -------------------------------------------------------------------------- *)
  (* --- SCOPING RULES                                                      --- *)
  (* -------------------------------------------------------------------------- *)

  val scope : t_env -> varinfo list -> scope -> t_prop -> t_prop
  val close : t_env -> t_prop -> t_prop

  (* -------------------------------------------------------------------------- *)
  (* --- FROM                                                               --- *)
  (* -------------------------------------------------------------------------- *)

  (** build [p => alpha(p)] for functional dependencies verification. *)
  val build_prop_of_from : t_env -> WpPropId.pred_info list -> t_prop -> t_prop

end
end
module Context : sig
# 1 "./Context.mli"
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

(** Current Loc *)

val with_current_loc : Cil_types.location -> ('a -> 'b) -> 'a -> 'b

(** Contextual Values *)

type 'a value

val create : ?default:'a -> string -> 'a value
(** Creates a new context with name *)

val defined : 'a value -> bool
(** The current value is defined. *)

val get : 'a value -> 'a
(** Retrieves the current value of the context.
    Raise an exception if not bound. *)

val get_opt : 'a value -> 'a option
(** Retrieves the current value of the context.
    Return [None] if not bound. *)

val set : 'a value -> 'a -> unit
(** Define the current value. Previous one is lost *)

val update : 'a value -> ('a -> 'a) -> unit
(** Modification of the current value *)

val bind : 'a value -> 'a -> ('b -> 'c) -> 'b -> 'c
(** Performs the job with local context bound to local value. *)

val free : 'a value -> ('b -> 'c) -> 'b -> 'c
(** Performs the job with local context cleared. *)

val clear : 'a value -> unit
(** Clear the current value. *)

val push : 'a value -> 'a -> 'a option
val pop : 'a value -> 'a option -> unit

val name : 'a value -> string

val register : (unit -> unit) -> unit
(** Register a global configure, to be executed once per project/ast. *)

val configure : unit -> unit
(** Apply global configure hooks, once per project/ast. *)
end
module Warning : sig
# 1 "./Warning.mli"
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

(** Contextual Errors *)

exception Error of string * string (** Source, Reason *)

val error : ?source:string -> ('a,Format.formatter,unit,'b) format4 -> 'a

(** Warning Manager *)

type t = {
  loc : Filepath.position ;
  severe : bool ;
  source : string ;
  reason : string ;
  effect : string ;
}

val compare : t -> t -> int
val pretty : Format.formatter -> t -> unit

module Set : FCSet.S with type elt = t
module Map : FCMap.S with type key = t

val severe : Set.t -> bool

type context

val context : ?source:string -> unit -> context
val flush : context -> Set.t
val add : t -> unit

val create : ?log:bool -> ?severe:bool -> ?source:string -> effect:string ->
  ('a,Format.formatter,unit,t) format4 -> 'a

val emit : ?severe:bool -> ?source:string -> effect:string ->
  ('a,Format.formatter,unit) format -> 'a
(** Emit a warning in current context.
    Defaults: [severe=true], [source="wp"]. *)

val handle : ?severe:bool -> effect:string -> handler:('a -> 'b) -> ('a -> 'b) -> 'a -> 'b
(** Handle the error and emit a warning with specified severity and effect
    if a context has been set.
    Otherwise, a WP-fatal error is raised instead.
    Default for [severe] is false. *)

type 'a outcome =
  | Result of Set.t * 'a
  | Failed of Set.t

val catch : ?source:string -> ?severe:bool -> effect:string -> ('a -> 'b) -> 'a -> 'b outcome
(** Set up a context for the job. If non-handled errors are raised,
    then a warning is emitted with specified severity and effect.
    Default for [severe] is [true]. *)
end
module WpContext : sig
# 1 "./wpContext.mli"
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

(** Model Registration *)

type model
type scope = Global | Kf of Kernel_function.t
type tuning = (unit -> unit)
type hypotheses = unit -> MemoryContext.clause list

val register :
  id:string ->
  ?descr:string ->
  ?tuning:tuning list ->
  ?hypotheses:hypotheses ->
  unit -> model

val get_descr : model -> string
val get_emitter : model -> Emitter.t

val compute_hypotheses : model -> Kernel_function.t -> MemoryContext.clause list

type context = model * scope
type t = context

module S :
sig
  type t = context
  val id : t -> string
  val hash : t -> int
  val equal : t -> t -> bool
  val compare : t -> t -> int
end

module MODEL :
sig
  type t = model
  val id : t -> string
  val descr : t -> string
  val hash : t -> int
  val equal : t -> t -> bool
  val compare : t -> t -> int
  val repr : t
end

module SCOPE :
sig
  type t = scope
  val id : t -> string
  val hash : t -> int
  val equal : t -> t -> bool
  val compare : t -> t -> int
end

val is_defined : unit -> bool
val on_context : context -> ('a -> 'b) -> 'a -> 'b
val get_model : unit -> model
val get_scope : unit -> scope
val get_context : unit -> context

val directory : unit -> string (** Current model in ["-wp-out"] directory *)

module type Entries =
sig
  type key
  type data
  val name : string
  val compare : key -> key -> int
  val pretty : Format.formatter -> key -> unit
end

module type Registry =
sig

  module E : Entries
  type key = E.key
  type data = E.data

  val id : basename:string -> key -> string
  val mem : key -> bool
  val find : key -> data
  val get : key -> data option
  val clear : unit -> unit
  val remove : key -> unit
  val define : key -> data -> unit
  (** no redefinition ; circularity protected *)
  val update : key -> data -> unit
  (** set current value, with no protection *)
  val memoize : (key -> data) -> key -> data
  (** with circularity protection *)
  val compile : (key -> data) -> key -> unit
  (** with circularity protection *)

  val callback : (key -> data -> unit) -> unit

  val iter : (key -> data -> unit) -> unit
  val iter_sorted : (key -> data -> unit) -> unit
end

module Index(E : Entries) : Registry with module E = E
(** projectified, depend on the model, not serialized *)

module Static(E : Entries) : Registry with module E = E
(** projectified, independent from the model, not serialized *)

module type Key =
sig
  type t
  val compare : t -> t -> int
  val pretty : Format.formatter -> t -> unit
end

module type Data =
sig
  type key
  type data
  val name : string
  val compile : key -> data
end

module type IData =
sig
  type key
  type data
  val name : string
  val basename : key -> string
  val compile : key -> string -> data
end

module type Generator =
sig
  type key
  type data
  val get : key -> data
  val mem : key -> bool
  val clear : unit -> unit
  val remove : key -> unit
end

(** projectified, depend on the model, not serialized *)
module Generator(K : Key)(D : Data with type key = K.t) : Generator
  with type key = D.key
   and type data = D.data

(** projectified, independent from the model, not serialized *)
module StaticGenerator(K : Key)(D : Data with type key = K.t) : Generator
  with type key = D.key
   and type data = D.data

(** projectified, depend on the model, not serialized *)
module GeneratorID(K : Key)(D : IData with type key = K.t) : Generator
  with type key = D.key
   and type data = D.data

(** projectified, independent from the model, not serialized *)
module StaticGeneratorID(K : Key)(D : IData with type key = K.t) : Generator
  with type key = D.key
   and type data = D.data
end
module Lang : sig
# 1 "./Lang.mli"
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

open Cil_types
open Ctypes
open Qed
open Qed.Logic

(** Logic Language based on Qed *)

(** {2 Library} *)

type library = string

(** Name for external prover.

    In case a Qed.Engine.link is used, [F_subst] patterns
    are not supported for Why-3. *)
type 'a infoprover = {
  altergo: 'a;
  why3   : 'a;
  coq    : 'a;
}
(** generic way to have different informations for the provers *)

val infoprover: 'a -> 'a infoprover
(** same information for all the provers *)
(** {2 Naming} Unique identifiers. *)

val comp_id  : compinfo -> string
val field_id : fieldinfo -> string
val type_id  : logic_type_info -> string
val logic_id : logic_info -> string
val ctor_id  : logic_ctor_info -> string
val lemma_id : string -> string

(** {2 Symbols} *)

type adt = private (** A type is never registered in a Definition.t *)
  | Mtype of mdt (** External type *)
  | Mrecord of mdt * fields (** External record-type *)
  | Atype of logic_type_info (** Logic Type *)
  | Comp of compinfo (** C-code struct or union *)
and mdt = string extern (** name to print to the provers *)
and 'a extern = {
  ext_id     : int;
  ext_link   : 'a infoprover;
  ext_library : library; (** a library which it depends on *)
  ext_debug  : string; (** just for printing during debugging *)
}
and fields = { mutable fields : field list }
and field =
  | Mfield of mdt * fields * string * tau
  | Cfield of fieldinfo
and tau = (field,adt) Logic.datatype


type lfun =
  | ACSL of Cil_types.logic_info (** Registered in Definition.t,
                                     only  *)
  | CTOR of Cil_types.logic_ctor_info (** Not registered in Definition.t
                                          directly converted/printed *)
  | Model of model (** *)

and model = {
  m_category : lfun category ;
  m_params : sort list ;
  m_result : sort ;
  m_typeof : tau option list -> tau ;
  m_source : source ;
}

and source =
  | Generated of WpContext.context option * string
  | Extern of Engine.link extern

val mem_builtin_type : name:string -> bool
val set_builtin_type : name:string -> link:string infoprover -> library:string -> unit
val get_builtin_type : name:string -> link:string infoprover -> library:string -> adt
val is_builtin : logic_type_info -> bool
val is_builtin_type : name:string -> tau -> bool
val datatype : library:string -> string -> adt
val record :
  link:string infoprover -> library:string -> (string * tau) list -> adt
val atype : logic_type_info -> adt
val comp : compinfo -> adt
val field : adt -> string -> field
val fields_of_adt : adt -> field list
val fields_of_tau : tau -> field list
val fields_of_field : field -> field list

type balance = Nary | Left | Right

val extern_s :
  library:library ->
  ?link:(Engine.link infoprover) ->
  ?category:lfun category ->
  ?params:sort list ->
  ?sort:sort ->
  ?result:tau ->
  ?typecheck:(tau option list -> tau) ->
  string -> lfun

val extern_f :
  library:library ->
  ?link:(Engine.link infoprover) ->
  ?balance:balance ->
  ?category:lfun category ->
  ?params:sort list ->
  ?sort:sort ->
  ?result:tau ->
  ?typecheck:(tau option list -> tau) ->
  ('a,Format.formatter,unit,lfun) format4 -> 'a
(** balance just give a default when link is not specified *)

val extern_p :
  library:library ->
  ?bool:string ->
  ?prop:string ->
  ?link:Engine.link infoprover ->
  ?params:sort list ->
  unit -> lfun

val extern_fp : library:library -> ?params:sort list ->
  ?link:string infoprover -> string -> lfun

val generated_f : ?context:bool -> ?category:lfun category ->
  ?params:sort list -> ?sort:sort -> ?result:tau ->
  ('a,Format.formatter,unit,lfun) format4 -> 'a

val generated_p : ?context:bool -> string -> lfun

(** {2 Sorting and Typing} *)

val tau_of_comp : compinfo -> tau
val tau_of_object : c_object -> tau
val tau_of_ctype : typ -> tau
val tau_of_ltype : logic_type -> tau
val tau_of_return : logic_info -> tau
val tau_of_lfun : lfun -> tau option list -> tau
val tau_of_field : field -> tau
val tau_of_record : field -> tau

val t_int : tau
val t_real : tau
val t_bool : tau
val t_prop : tau
val t_addr : unit -> tau (** pointer on Void *)
val t_array : tau -> tau
val t_farray : tau -> tau -> tau
val t_datatype : adt -> tau list -> tau

val pointer : (typ -> tau) Context.value (** type of pointers *)
val floats : (c_float -> tau) Context.value (** type of floats *)
val poly : string list Context.value (** polymorphism *)
val parameters : (lfun -> sort list) -> unit (** definitions *)

val name_of_lfun : lfun -> string
val name_of_field : field -> string

(** {2 Logic Formulae} *)

module ADT : Logic.Data with type t = adt
module Field : Logic.Field with type t = field
module Fun : Logic.Function with type t = lfun

class virtual idprinting :
  object
    method virtual sanitize : string -> string

    method virtual infoprover : 'a. 'a infoprover -> 'a
    (** Specify the field to use in an infoprover *)

    method sanitize_type : string -> string
    (** Defaults to [self#sanitize] *)

    method sanitize_field : string -> string
    (** Defulats to [self#sanitize] *)

    method sanitize_fun : string -> string
    (** Defulats to [self#sanitize] *)

    method datatype : ADT.t   -> string
    method field    : Field.t -> string
    method link     : Fun.t   -> Engine.link
  end

module F :
sig

  module QED : Logic.Term with module ADT = ADT
                           and module Field = Field
                           and module Fun = Fun

  (** {3 Types and Variables} *)

  type var = QED.var
  type tau = QED.tau
  type pool = QED.pool
  module Tau = QED.Tau
  module Var = QED.Var
  module Vars : Qed.Idxset.S with type elt = var
  module Vmap : Qed.Idxmap.S with type key = var

  val pool : ?copy:pool -> unit -> pool
  val fresh : pool -> ?basename:string -> tau -> var
  val alpha : pool -> var -> var
  val add_var : pool -> var -> unit
  val add_vars : pool -> Vars.t -> unit

  val tau_of_var : var -> tau

  (** {3 Expressions} *)

  type term = QED.term
  type record = (field * term) list

  val hash : term -> int (** Constant time *)
  val equal : term -> term -> bool (** Same as [==] *)
  val compare : term -> term -> int

  module Tset : Qed.Idxset.S with type elt = term
  module Tmap : Qed.Idxmap.S with type key = term

  type unop = term -> term
  type binop = term -> term -> term

  val e_zero : term
  val e_one : term
  val e_minus_one : term
  val e_minus_one_real : term
  val e_one_real : term
  val e_zero_real : term

  val constant : term -> term

  val e_fact : int -> term -> term

  val e_int64 : int64 -> term
  val e_bigint : Integer.t -> term
  val e_float : float -> term
  val e_setfield : term -> field -> term -> term
  val e_range : term -> term -> term (** e_range a b = b+1-a *)
  val is_zero : term -> bool

  val e_true : term
  val e_false : term
  val e_bool : bool -> term
  val e_literal : bool -> term -> term
  val e_int : int -> term
  val e_zint : Z.t -> term
  val e_real : Q.t -> term
  val e_var : var -> term
  val e_opp : term -> term
  val e_times : Z.t -> term -> term
  val e_sum : term list -> term
  val e_prod : term list -> term
  val e_add : term -> term -> term
  val e_sub : term -> term -> term
  val e_mul : term -> term -> term
  val e_div : term -> term -> term
  val e_mod : term -> term -> term
  val e_eq  : term -> term -> term
  val e_neq : term -> term -> term
  val e_leq : term -> term -> term
  val e_lt  : term -> term -> term
  val e_imply : term list -> term -> term
  val e_equiv : term -> term -> term
  val e_and   : term list -> term
  val e_or    : term list -> term
  val e_not   : term -> term
  val e_if    : term -> term -> term -> term
  val e_const : tau -> term -> term
  val e_get   : term -> term -> term
  val e_set   : term -> term -> term -> term
  val e_getfield : term -> Field.t -> term
  val e_record : record -> term
  val e_fun : ?result:tau -> Fun.t -> term list -> term
  val e_bind : binder -> var -> term -> term

  val e_open : pool:pool -> ?forall:bool -> ?exists:bool -> ?lambda:bool ->
    term -> (binder * var) list * term
  (** Open all the specified binders (flags default to `true`, so all
      consecutive top most binders are opened by default).
      The pool must contain all free variables of the term. *)

  val e_close : (binder * var) list -> term -> term
  (** Closes all specified binders *)

  (** {3 Predicates} *)

  type pred
  type cmp = term -> term -> pred
  type operator = pred -> pred -> pred
  module Pmap : Qed.Idxmap.S with type key = pred
  module Pset : Qed.Idxset.S with type elt = pred

  val p_true : pred
  val p_false : pred

  val p_equal : term -> term -> pred
  val p_equals : (term * term) list -> pred list
  val p_neq : term -> term -> pred
  val p_leq : term -> term -> pred
  val p_lt : term -> term -> pred
  val p_positive : term -> pred

  val is_ptrue : pred -> Logic.maybe
  val is_pfalse : pred -> Logic.maybe
  val is_equal : term -> term -> Logic.maybe
  val eqp : pred -> pred -> bool
  val comparep : pred -> pred -> int

  val p_bool : term -> pred
  val e_prop : pred -> term
  val p_bools : term list -> pred list
  val e_props : pred list -> term list
  val lift : (term -> term) -> pred -> pred

  val p_not : pred -> pred
  val p_and : pred -> pred -> pred
  val p_or  : pred -> pred -> pred
  val p_imply : pred -> pred -> pred
  val p_equiv : pred -> pred -> pred
  val p_hyps : pred list -> pred -> pred
  val p_if : pred -> pred -> pred -> pred

  val p_conj : pred list -> pred
  val p_disj : pred list -> pred

  val p_any : ('a -> pred) -> 'a list -> pred
  val p_all : ('a -> pred) -> 'a list -> pred

  val p_call : lfun -> term list -> pred

  val p_forall : var list -> pred -> pred
  val p_exists : var list -> pred -> pred
  val p_bind : binder -> var -> pred -> pred

  type sigma

  module Subst :
  sig
    val get : sigma -> term -> term
    val add : sigma -> term -> term -> unit
    val add_map : sigma -> term Tmap.t -> unit
    val add_fun : sigma -> (term -> term) -> unit
    val add_filter : sigma -> (term -> bool) -> unit
  end

  val e_subst : sigma -> term -> term
  val p_subst : sigma -> pred -> pred
  val p_subst_var : var -> term -> pred -> pred

  val e_vars : term -> var list (** Sorted *)
  val p_vars : pred -> var list (** Sorted *)

  val p_close : pred -> pred (** Quantify over (sorted) free variables *)

  val pred_transform: pred -> pred
  val add_node: pred -> String.t -> unit
  val add_node_eq : var -> var -> unit
  val update_namespace_term: pred -> unit
  val dag_copy: unit -> unit
  val dag_copy_replace: unit -> unit
  val update_term_name : pred list -> pred list
  val add_name_fun: String.t -> unit

  val pp_tau : Format.formatter -> tau -> unit
  val pp_var : Format.formatter -> var -> unit
  val pp_vars : Format.formatter -> Vars.t -> unit
  val pp_term : Format.formatter -> term -> unit
  val pp_pred : Format.formatter -> pred -> unit

  val debugp : Format.formatter -> pred -> unit

  type env
  val context_pp : env Context.value
  (** Context used by pp_term, pp_pred, pp_var, ppvars for printing
      the term. Allows to keep the same disambiguation. *)

  type marks = QED.marks

  val env : Vars.t -> env
  val marker : env -> marks
  val mark_e : marks -> term -> unit
  val mark_p : marks -> pred -> unit
  (** Returns a list of terms to be shared among all {i shared} or {i
      marked} subterms.  The order of terms is consistent with
      definition order: head terms might be used in tail ones. *)
  val defs : marks -> term list
  val define : (env -> string -> term -> unit) -> env -> marks -> env
  val pp_eterm : env -> Format.formatter -> term -> unit
  val pp_epred : env -> Format.formatter -> pred -> unit

  val p_expr : pred -> pred QED.expression
  val e_expr : pred -> term QED.expression

  (* val p_iter : (pred -> unit) -> (term -> unit) -> pred -> unit *)

  (** {3 Binders} *)

  val lc_closed : term -> bool
  val lc_iter : (term -> unit) -> term -> unit (* TODO: to remove *)

  (** {3 Utilities} *)

  val decide   : term -> bool (** Return [true] if and only the term is [e_true]. Constant time. *)
  val basename : term -> string
  val is_true  : term -> maybe (** Constant time. *)
  val is_false : term -> maybe (** Constant time. *)
  val is_prop  : term -> bool (** Boolean or Property *)
  val is_int   : term -> bool (** Integer sort *)
  val is_real  : term -> bool (** Real sort *)
  val is_arith : term -> bool (** Integer or Real sort *)

  val is_closed : term -> bool (** No bound variables *)
  val is_simple : term -> bool (** Constants, variables, functions of arity 0 *)
  val is_atomic : term -> bool (** Constants and variables *)
  val is_primitive : term -> bool (** Constants only *)
  val is_neutral : Fun.t -> term -> bool
  val is_absorbant : Fun.t -> term -> bool
  val record_with : record -> (term * record) option

  val are_equal : term -> term -> maybe (** Computes equality *)
  val eval_eq   : term -> term -> bool  (** Same as [are_equal] is [Yes] *)
  val eval_neq  : term -> term -> bool  (** Same as [are_equal] is [No]  *)
  val eval_lt   : term -> term -> bool  (** Same as [e_lt] is [e_true] *)
  val eval_leq  : term -> term -> bool  (** Same as [e_leq] is [e_true]  *)

  val repr : term -> QED.repr (** Constant time *)
  val sort : term -> Logic.sort (** Constant time *)
  val vars : term -> Vars.t (** Constant time *)
  val varsp : pred -> Vars.t (** Constant time *)
  val occurs : var -> term -> bool
  val occursp : var -> pred -> bool
  val intersect : term -> term -> bool
  val intersectp : pred -> pred -> bool
  val is_subterm : term -> term -> bool

  (** Try to extract a type of term.
      Parameterized by optional extractors for field and functions.
      Extractors may raise [Not_found] ; however, they are only used when
      the provided kinds for fields and functions are not precise enough.
      @param field type of a field value
      @param record type of the record containing a field
      @param call type of the values returned by the function
      @raise Not_found if no type is found. *)
  val typeof :
    ?field:(Field.t -> tau) ->
    ?record:(Field.t -> tau) ->
    ?call:(Fun.t -> tau option list -> tau) ->
    term -> tau

  (** {3 Builtins}

      The functions below register simplifiers for function [f]. The computation
      code may raise [Not_found], in which case the symbol is not interpreted.

      If [f] is an operator with algebraic rules (see type [operator]), the
      children are normalized {i before} builtin call.

      Highest priority is [0]. Recursive calls must be performed on strictly
      smaller terms. *)

  val set_builtin : lfun -> (term list -> term) -> unit
  val set_builtin_get : lfun -> (term list -> tau option -> term-> term) -> unit
  val set_builtin_1 : lfun -> unop -> unit
  val set_builtin_2 : lfun -> binop -> unit
  val set_builtin_2' : lfun -> (term -> term -> tau option -> term) -> unit
  val set_builtin_eq : lfun -> binop -> unit
  val set_builtin_leq : lfun -> binop -> unit
  val set_builtin_eqp : lfun -> cmp -> unit

  val release : unit -> unit (** Empty local caches *)

end


module N: sig
  (** simpler notation for writing {!F.term} and {F.pred} *)

  val ( + ): F.binop (** {! F.p_add } *)
  val ( - ): F.binop (** {! F.p_sub } *)
  val ( ~- ): F.unop (** [fun x -> p_sub 0 x] *)
  val ( * ): F.binop (** {! F.p_mul} *)
  val ( / ): F.binop (** {! F.p_div} *)
  val ( mod ): F.binop (** {! F.p_mod} *)

  val ( = ): F.cmp (** {! F.p_equal} *)
  val ( < ): F.cmp (** {! F.p_lt} *)
  val ( > ): F.cmp (** {! F.p_lt} with inversed argument *)
  val ( <= ): F.cmp (** {! F.p_leq } *)
  val ( >= ): F.cmp (** {! F.p_leq } with inversed argument *)
  val ( <> ): F.cmp (** {! F.p_neq } *)

  val ( && ): F.operator (** {! F.p_and } *)
  val ( || ): F.operator (** {! F.p_or } *)
  val not: F.pred -> F.pred (** {! F.p_not } *)

  val ( $ ): ?result:tau -> lfun -> F.term list -> F.term (** {! F.e_fun } *)
  val ( $$ ): lfun -> F.term list -> F.pred (** {! F.p_call } *)
end


(** {2 Fresh Variables and Constraints} *)

open F

type gamma
val new_pool : ?copy:F.pool -> ?vars:Vars.t -> unit -> pool
val new_gamma : ?copy:gamma -> unit -> gamma

val local : ?pool:pool -> ?vars:Vars.t -> ?gamma:gamma -> ('a -> 'b) -> 'a -> 'b

val freshvar : ?basename:string -> tau -> var
val freshen : var -> var
val assume : pred -> unit
val without_assume : ('a -> 'b) -> 'a -> 'b
val epsilon : ?basename:string -> tau -> (term -> pred) -> term
val hypotheses : gamma -> pred list
val variables : gamma -> var list

val get_pool : unit -> pool
val get_gamma : unit -> gamma
val has_gamma : unit -> bool
val get_hypotheses : unit -> pred list
val get_variables : unit -> var list

(** {2 Substitutions} *)

val sigma : unit -> F.sigma (** uses current pool *)
val alpha : unit -> F.sigma (** freshen all variables *)
val subst : F.var list -> F.term list -> F.sigma (** replace variables *)

val e_subst : (term -> term) -> term -> term (** uses current pool *)
val p_subst : (term -> term) -> pred -> pred (** uses current pool *)

(** {2 Simplifiers} *)

exception Contradiction

val is_literal: F.term -> bool
val iter_consequence_literals: (F.term -> unit) -> F.term -> unit
(** [iter_consequence_literals assume_from_litteral hypothesis] applies
    the function [assume_from_litteral] on literals that are a consequence of the [hypothesis]
    (i.e. in the hypothesis [not (A && (B || C) ==> D)], only [A] and [not D] are
    considered as consequence literals). *)

class type simplifier =
  object
    method name : string
    method copy : simplifier
    method assume : F.pred -> unit
    (** Assumes the hypothesis *)
    method target : F.pred -> unit
    (** Give the predicate that will be simplified later *)
    method fixpoint : unit
    (** Called after assuming hypothesis and knowing the goal *)
    method infer : F.pred list
    (** Add new hypotheses implied by the original hypothesis. *)

    method simplify_exp : F.term -> F.term
    (** Currently simplify an expression. *)
    method simplify_hyp : F.pred -> F.pred
    (** Currently simplify an hypothesis before assuming it. In any
        case must return a weaker formula. *)
    method simplify_branch : F.pred -> F.pred
    (** Currently simplify a branch condition. In any case must return an
        equivalent formula. *)
    method simplify_goal : F.pred -> F.pred
    (** Simplify the goal. In any case must return a stronger formula. *)
  end

(* -------------------------------------------------------------------------- *)

(** For why3_api but circular dependency *)
module For_export : sig

  type specific_equality = {
    for_tau:(tau -> bool);
    mk_new_eq:F.binop;
  }

  val rebuild : ?cache:term Tmap.t -> term -> term * term Tmap.t

  val set_builtin : Fun.t -> (term list -> term) -> unit
  val set_builtin' : Fun.t -> (term list -> tau option -> term) -> unit

  val set_builtin_eq : Fun.t -> (term -> term -> term) -> unit
  val set_builtin_leq : Fun.t -> (term -> term -> term) -> unit

  val in_state: ('a -> 'b) -> 'a -> 'b

end
end
module Repr : sig
# 1 "./Repr.mli"
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

(** {2 Term & Predicate Introspection} *)

type tau = Lang.F.tau
type var = Lang.F.var
type field = Lang.field
type lfun = Lang.lfun
type term = Lang.F.term
type pred = Lang.F.pred

type repr =
  | True
  | False
  | And of term list
  | Or of term list
  | Not of term
  | Imply of term list * term
  | If of term * term * term
  | Var of var
  | Int of Z.t
  | Real of Q.t
  | Add of term list
  | Mul of term list
  | Div of term * term
  | Mod of term * term
  | Eq of term * term
  | Neq of term * term
  | Lt of term * term
  | Leq of term * term
  | Times of Z.t * term
  | Call of lfun * term list
  | Field of term * field
  | Record of (field * term) list
  | Cst of tau * term
  | Get of term * term
  | Set of term * term * term
  | HigherOrder (** See Lang.F.e_open and Lang.F.e_close *)

val term : term -> repr
val pred : pred -> repr

val lfun : lfun -> string
val field : field -> string

(* -------------------------------------------------------------------------- *)
end
module Passive : sig
# 1 "./Passive.mli"
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

open Lang.F

(** Passive Forms *)
type binding =
  | Bind of var * var (* fresh , bound *)
  | Join of var * var (* left, right *)

type t = binding list

val empty : t
val is_empty : t -> bool
val union : t -> t -> t
val bind : fresh:var -> bound:var -> t -> t
val join : var -> var -> t -> t
val conditions : t -> (var -> bool) -> pred list
val apply : t -> pred -> pred



val iter : (binding -> unit) -> t -> unit
val pretty : Format.formatter -> t -> unit
end
module Splitter : sig
# 1 "./Splitter.mli"
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

open Cil_types

type tag =
  | MARK of stmt
  | THEN of stmt
  | ELSE of stmt
  | CALL of stmt * kernel_function
  | CASE of stmt * int64 list
  | DEFAULT of stmt
  | ASSERT of identified_predicate * int * int (* part / Npart *)

val loc : tag -> location
val pretty : Format.formatter -> tag -> unit

val mark : stmt -> tag
val if_then : stmt -> tag
val if_else : stmt -> tag
val switch_cases : stmt -> int64 list -> tag
val switch_default : stmt -> tag
val cases : identified_predicate -> (tag * predicate) list option
val call : stmt -> kernel_function -> tag

type 'a t

val empty : 'a t
val singleton : 'a -> 'a t
val group : tag -> ('a list -> 'a) -> 'a t -> 'a t

val union : ('a -> 'a -> 'a) -> 'a t -> 'a t -> 'a t
val merge :
  left:('a -> 'c) ->
  both:('a -> 'b -> 'c) ->
  right:('b -> 'c) ->
  'a t -> 'b t -> 'c t

val merge_all : ('a list -> 'a) -> 'a t list -> 'a t

val length : 'a t -> int

val map : ('a -> 'b) -> 'a t -> 'b t
val iter : (tag list -> 'a -> unit) -> 'a t -> unit
val fold : (tag list -> 'a -> 'b -> 'b) -> 'a t -> 'b -> 'b

val exists : ('a -> bool) -> 'a t -> bool
val for_all : ('a -> bool) -> 'a t -> bool
val filter : ('a -> bool) -> 'a t -> 'a t
end
module LogicBuiltins : sig
# 1 "./LogicBuiltins.mli"
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

(* -------------------------------------------------------------------------- *)
(* --- Linker for ACSL Builtins                                           --- *)
(* -------------------------------------------------------------------------- *)

open Cil_types
open Lang

type category = Lang.lfun Qed.Logic.category

type kind =
  | Z                   (** integer *)
  | R                   (** real *)
  | I of Ctypes.c_int   (** C-ints *)
  | F of Ctypes.c_float (** C-floats *)
  | A                   (** Abstract Data *)

val kind_of_tau : tau -> kind

(** Add a new builtin. This builtin will be shared with all created drivers *)
val add_builtin : string -> kind list -> lfun -> unit

type driver
val driver: driver Context.value

val create: id:string -> ?descr:string -> ?includes:string list -> unit -> driver
(** Create a new driver. leave the context empty. *)

val init: id:string -> ?descr:string -> ?includes:string list -> unit -> unit
(** Reset the context to a newly created driver *)

val id : driver -> string
val descr : driver -> string
val is_default : driver -> bool
val compare : driver -> driver -> int

val find_lib: string -> string
(** find a file in the includes of the current drivers *)

val dependencies : string -> string list
(** Of external theories. Raises Not_found if undefined *)

val add_library : string -> string list -> unit
(** Add a new library or update the dependencies of an existing one *)

val add_alias : source:Filepath.position -> string -> kind list -> alias:string -> unit -> unit

val add_type : source:Filepath.position -> string -> library:string ->
  ?link:string infoprover -> unit -> unit

val add_ctor : source:Filepath.position -> string -> kind list ->
  library:string -> link:Qed.Engine.link infoprover -> unit -> unit

val add_logic : source:Filepath.position -> kind -> string -> kind list ->
  library:string -> ?category:category -> link:Qed.Engine.link infoprover ->
  unit -> unit

val add_predicate : source:Filepath.position -> string -> kind list ->
  library:string -> link:string infoprover ->
  unit -> unit

val add_option :
  driver_dir:string -> string -> string -> library:string -> string -> unit
(** add a value to an option (group, name) *)

val set_option :
  driver_dir:string -> string -> string -> library:string -> string -> unit
(** reset and add a value to an option (group, name) *)

type doption

type sanitizer = (driver_dir:string -> string -> string)

val create_option: sanitizer:sanitizer -> string -> string -> doption
(** [add_option_sanitizer ~driver_dir group name]
    add a sanitizer for group [group] and option [name] *)

val get_option : doption -> library:string -> string list
(** return the values of option (group, name),
    return the empty list if not set *)

type builtin =
  | ACSLDEF
  | LFUN of lfun
  | HACK of (F.term list  -> F.term)

val logic : logic_info -> builtin
val ctor : logic_ctor_info -> builtin
val constant : string -> builtin
val lookup : string -> kind list -> builtin

(** Replace a logic definition or predicate by a built-in function.
    The LogicSemantics compilers will replace `Pcall` and `Tcall` instances
    of this symbol with the provided Qed function on terms. *)
val hack : string -> (F.term list -> F.term) -> unit

val dump : unit -> unit
end
module Definitions : sig
# 1 "./Definitions.mli"
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

open LogicUsage
open Cil_types
open Ctypes
open Lang
open Lang.F

type cluster

val dummy : unit -> cluster
val cluster : id:string -> ?title:string -> ?position:Filepath.position -> unit -> cluster
val axiomatic : axiomatic -> cluster
val section : logic_section -> cluster
val compinfo : compinfo -> cluster
val matrix : c_object -> cluster

val cluster_id : cluster -> string (** Unique *)
val cluster_title : cluster -> string
val cluster_position : cluster -> Filepath.position option
val cluster_age : cluster -> int
val cluster_compare : cluster -> cluster -> int
val pp_cluster : Format.formatter -> cluster -> unit
val iter : (cluster -> unit) -> unit

type trigger = (var,lfun) Qed.Engine.ftrigger
type typedef = (tau,field,lfun) Qed.Engine.ftypedef

type dlemma = {
  l_name  : string ;
  l_cluster : cluster ;
  l_assumed : bool ;
  l_types : int ;
  l_forall : var list ;
  l_triggers : trigger list list ; (** OR of AND-triggers *)
  l_lemma : pred ;
}

type definition =
  | Logic of tau
  | Function of tau * recursion * term
  | Predicate of recursion * pred
  | Inductive of dlemma list

and recursion = Def | Rec

type dfun = {
  d_lfun   : lfun ;
  d_cluster : cluster ;
  d_types  : int ;
  d_params : var list ;
  d_definition : definition ;
}

module Trigger :
sig
  val of_term : term -> trigger
  val of_pred : pred -> trigger
  val vars : trigger -> Vars.t
end

val find_symbol : lfun -> dfun (** @raise Not_found if symbol is not compiled (yet) *)
val define_symbol : dfun -> unit
val update_symbol : dfun -> unit

val find_name : string -> dlemma
val find_lemma : logic_lemma -> dlemma (** @raise Not_found if lemma is not compiled (yet) *)
val compile_lemma  : (logic_lemma -> dlemma) -> logic_lemma -> unit
val define_lemma  : dlemma -> unit
val define_type   : cluster -> logic_type_info -> unit

val call_fun : result:tau -> lfun -> (lfun -> dfun) -> term list -> term
val call_pred : lfun -> (lfun -> dfun) -> term list -> pred

type axioms = cluster * logic_lemma list

class virtual visitor : cluster ->
  object

    (** {2 Locality} *)

    method set_local : cluster -> unit
    method do_local : cluster -> bool

    (** {2 Visiting items} *)

    method vadt : ADT.t -> unit
    method vtype : logic_type_info -> unit
    method vcomp : compinfo -> unit
    method vfield : Field.t -> unit
    method vtau : tau -> unit
    method vparam : var -> unit
    method vterm : term -> unit
    method vpred : pred -> unit
    method vsymbol : lfun -> unit
    method vlemma : logic_lemma -> unit
    method vcluster : cluster -> unit
    method vlibrary : string -> unit
    method vgoal : axioms option -> F.pred -> unit
    method vtypes : unit (** Visit all typedefs *)
    method vsymbols : unit (** Visit all definitions *)
    method vlemmas : unit (** Visit all lemmas *)
    method vself : unit (** Visit all records, types, defs and lemmas *)

    (** {2 Visited definitions} *)

    method virtual section : string -> unit (** Comment *)
    method virtual on_library : string -> unit (** External library to import *)
    method virtual on_cluster : cluster -> unit (** Outer cluster to import *)
    method virtual on_type : logic_type_info -> typedef -> unit (** This local type must be defined *)
    method virtual on_comp : compinfo -> (field * tau) list -> unit (** This local compinfo must be defined *)
    method virtual on_dlemma : dlemma -> unit (** This local lemma must be defined *)
    method virtual on_dfun : dfun -> unit (** This local function must be defined *)

  end
end
module Cint : sig
# 1 "./Cint.mli"
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

(* -------------------------------------------------------------------------- *)
(** Integer Arithmetic Model *)
(* -------------------------------------------------------------------------- *)

open Ctypes
open Lang
open Lang.F

val of_real : c_int -> unop
val convert : c_int -> unop (** Independent from model *)

val to_integer : unop
val of_integer : c_int -> unop

val to_cint : lfun -> c_int (** Raises [Not_found] if not. *)
val is_cint : lfun -> c_int (** Raises [Not_found] if not. *)

type model = Natural | Machine
val configure : model -> unit
val current : unit -> model

val range : c_int -> term -> pred (** Dependent on model *)
val downcast : c_int -> unop (** Dependent on model *)

val iopp : c_int -> unop
val iadd : c_int -> binop
val isub : c_int -> binop
val imul : c_int -> binop
val idiv : c_int -> binop
val imod : c_int -> binop

val bnot : c_int -> unop
val band : c_int -> binop
val bxor : c_int -> binop
val bor  : c_int -> binop
val blsl : c_int -> binop
val blsr : c_int -> binop

val l_not : unop
val l_and : binop
val l_xor : binop
val l_or  : binop
val l_lsl : binop
val l_lsr : binop

val f_lnot : lfun
val f_land : lfun
val f_lxor : lfun
val f_lor  : lfun
val f_lsl  : lfun
val f_lsr  : lfun

val f_bitwised : lfun list (** All except f_bit_positive *)

(** Simplifiers *)

val is_cint_simplifier: simplifier
(** Remove the [is_cint] in formulas that are
    redundant with other conditions. *)

val mask_simplifier: simplifier

val is_positive_or_null: term -> bool
end
module Cfloat : sig
# 1 "./Cfloat.mli"
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

(* -------------------------------------------------------------------------- *)
(** Floating Arithmetic Model *)
(* -------------------------------------------------------------------------- *)

open Ctypes
open Lang
open Lang.F

val f32 : adt
val f64 : adt

val t32 : tau
val t64 : tau

type model = Real | Float
val configure : model -> unit

val ftau : c_float -> tau (** model independant *)
val tau_of_float : c_float -> tau (** with respect to model *)

type op =
  | LT
  | EQ
  | LE
  | NE
  | NEG
  | ADD
  | MUL
  | DIV
  | REAL
  | ROUND
  | EXACT (** same as round, but argument is exact representation *)

val find : lfun -> op * c_float

val code_lit : c_float -> float -> string option -> term
val acsl_lit : Cil_types.logic_real -> term
val float_lit : c_float -> Q.t -> string
(** Returns a string literal in decimal notation (without suffix)
    that reparses to the same value (when added suffix). *)

val float_of_int : c_float -> unop
val float_of_real : c_float -> unop
val real_of_float : c_float -> unop

val fopp : c_float -> unop
val fadd : c_float -> binop
val fsub : c_float -> binop
val fmul : c_float -> binop
val fdiv : c_float -> binop

val flt : c_float -> cmp
val fle : c_float -> cmp
val feq : c_float -> cmp
val fneq : c_float -> cmp

val f_model : c_float -> lfun
val f_delta : c_float -> lfun
val f_epsilon : c_float -> lfun

val flt_of_real : c_float -> lfun
val real_of_flt : c_float -> lfun

val flt_add : c_float -> lfun
val flt_mul : c_float -> lfun
val flt_div : c_float -> lfun
val flt_neg : c_float -> lfun
end
module Vset : sig
# 1 "./Vset.mli"
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

open Lang.F

(** Logical Sets *)

type set = vset list
and vset =
  | Set of tau * term
  | Singleton of term
  | Range of term option * term option
  | Descr of var list * term * pred

val tau_of_set : tau -> tau

val vars : set -> Vars.t
val occurs : var -> set -> bool

val empty : set
val singleton : term -> set
val range : term option -> term option -> set
val union : set -> set -> set
val inter : term -> term -> term

val member : term -> set -> pred
val in_size : term -> int -> pred
val in_range : term -> term option -> term option -> pred
val sub_range : term -> term -> term option -> term option -> pred
val ordered : limit:bool -> strict:bool -> term option -> term option -> pred
(** - [limit]: result when either parameter is [None]
    - [strict]: if [true], comparison is [<] instead of [<=] *)

val is_empty : set -> pred
val equal : set -> set -> pred
val subset : set -> set -> pred
val disjoint : set -> set -> pred

val concretize : set -> term

val bound_shift : term option -> term -> term option
val bound_add : term option -> term option -> term option
val bound_sub : term option -> term option -> term option

(** {3 Pretty} *)

val pp_bound : Format.formatter -> term option -> unit
val pp_vset : Format.formatter -> vset -> unit
val pretty : Format.formatter -> set -> unit

(** {3 Mapping}
    These operations compute different kinds of [{f x y with x in A, y in B}].
*)

val map : (term -> term) -> set -> set
val map_opp : set -> set

(** {3 Lifting}
    These operations computes different sort of [{f x y with x in A, y in B}].
*)

val lift : (term -> term -> term) -> set -> set -> set
val lift_add : set -> set -> set
val lift_sub : set -> set -> set

val descr : vset -> var list * term * pred
end
module Cstring : sig
# 1 "./Cstring.mli"
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

(* -------------------------------------------------------------------------- *)
(* --- String Constants                                                   --- *)
(* -------------------------------------------------------------------------- *)

open Lang.F

type cst =
  | C_str of string (** String Literal *)
  | W_str of int64 list (** Wide String Literal *)

val pretty : Format.formatter -> cst -> unit

val str_len : cst -> term -> pred
(** Property defining the size of the string in bytes,
    with [\0] terminator included. *)

val str_val : cst -> term
(** The array containing the [char] of the constant *)

val str_id : cst -> int
(** Non-zero integer, unique for each different string literal *)

val char_at : cst -> term -> term

val cluster : unit -> Definitions.cluster
(** The cluster where all strings are defined. *)
end
module Sigs : sig
# 1 "./Sigs.ml"
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

(* -------------------------------------------------------------------------- *)
(** Common Types and Signatures *)
(* -------------------------------------------------------------------------- *)

open Cil_types
open Ctypes
open Lang.F
open Interpreted_automata

(* -------------------------------------------------------------------------- *)
(** {1 General Definitions} *)
(* -------------------------------------------------------------------------- *)

type 'a sequence = { pre : 'a ; post : 'a }

type 'a binder = { bind: 'b 'c. 'a -> ('b -> 'c) -> 'b -> 'c }

(** Oriented equality or arbitrary relation *)
type equation =
  | Set of term * term (** [Set(a,b)] is [a := b]. *)
  | Assert of pred

(** Access conditions *)
type acs =
  | RW (** Read-Write Access *)
  | RD (** Read-Only Access *)

(** Abstract location or concrete value *)
type 'a value =
  | Val of term
  | Loc of 'a

(** Contiguous set of locations *)
type 'a rloc =
  | Rloc of c_object * 'a
  | Rrange of 'a * c_object * term option * term option

(** Structured set of locations *)
type 'a sloc =
  | Sloc of 'a
  | Sarray of 'a * c_object * int (** full sized range (optimized assigns) *)
  | Srange of 'a * c_object * term option * term option
  | Sdescr of var list * 'a * pred

(** Typed set of locations *)
type 'a region = (c_object * 'a sloc) list

(** Logical values, locations, or sets of *)
type 'a logic =
  | Vexp of term
  | Vloc of 'a
  | Vset of Vset.set
  | Lset of 'a sloc list

(** Scope management for locals and formals *)
type scope = Enter | Leave

(** Container for the returned value of a function *)
type 'a result =
  | R_loc of 'a
  | R_var of var

(** Polarity of predicate compilation *)
type polarity = [ `Positive | `Negative | `NoPolarity ]

(** Frame Conditions.
    Consider a function [phi(m)] over memory [m],
    we want memories [m1,m2] and condition [p] such that
    [p(m1,m2) -> phi(m1) = phi(m2)].
    - [name] used for generating lemma
    - [triggers] for the lemma
    - [conditions] for the frame lemma to hold
    - [mem1,mem2] to two memories for which the lemma holds *)
type frame = string * Definitions.trigger list * pred list * term * term

(* -------------------------------------------------------------------------- *)
(** {1 Reversing Models}

    It is sometimes possible to reverse memory models abstractions
    into ACSL left-values via the definitions below. *)
(* -------------------------------------------------------------------------- *)

(** Reversed ACSL left-value *)
type s_lval = s_host * s_offset list

and s_host =
  | Mvar of varinfo (** Variable *)
  | Mmem of term    (** Pointed value *)
  | Mval of s_lval  (** Pointed value of another abstract left-value *)

and s_offset = Mfield of fieldinfo | Mindex of term

(** Reversed abstract value *)
type mval =
  | Mterm (** Not a state-related value *)
  | Maddr of s_lval (** The value is the address of an l-value in current memory *)
  | Mlval of s_lval (** The value is the value of an l-value in current memory *)
  | Mchunk of string (** The value is an abstract memory chunk (description) *)

(** Reversed update *)
type update = Mstore of s_lval * term
(** An update of the ACSL left-value with the given value *)

(* -------------------------------------------------------------------------- *)
(** {1 Memory Models} *)
(* -------------------------------------------------------------------------- *)

(** Memory Chunks.

    The concrete memory is partionned into a vector of abstract data.
    Each component of the partition is called a {i memory chunk} and
    holds an abstract representation of some part of the memory.

    Remark: memory chunks are not required to be independant from each other,
    provided the memory model implementation is consistent with the chosen
    representation. Conversely, a given object might be represented by
    several memory chunks. See {!Model.domain}.

*)
module type Chunk =
sig

  type t
  val self : string (** Chunk names, for pretty-printing. *)
  val hash : t -> int
  val equal : t -> t -> bool
  val compare : t -> t -> int
  val pretty : Format.formatter -> t -> unit

  val tau_of_chunk : t -> tau
  (** The type of data hold in a chunk. *)

  val basename_of_chunk : t -> string
  (** Used when generating fresh variables for a chunk. *)

  val is_framed : t -> bool
  (** Whether the chunk is local to a function call.

      Means the chunk is separated from anyother call side-effects.
      If [true], entails that a function assigning everything can not modify
      the chunk. Only used for optimisation, it would be safe to always
      return [false]. *)

end

(** Memory Environments.

    Represents the content of the memory, {i via} a vector of logic
    variables for each memory chunk.
*)
module type Sigma =
sig

  type chunk (** The type of memory chunks. *)
  module Chunk : Qed.Collection.S with type t = chunk

  (** Memory footprint. *)
  type domain = Chunk.Set.t

  (** Environment assigning logic variables to chunk.

      Memory chunk variables are assigned lazily. Hence, the vector is
      empty unless a chunk is accessed. Pay attention to this
      when you merge or havoc chunks.

      New chunks are generated from the context pool of {!Lang.freshvar}.
  *)
  type t

  val pretty : Format.formatter -> t -> unit
  (** For debugging purpose *)

  val create : unit -> t (** Initially empty environment. *)

  val mem : t -> chunk -> bool (** Whether a chunk has been assigned. *)
  val get : t -> chunk -> var (** Lazily get the variable for a chunk. *)
  val value : t -> chunk -> term (** Same as [Lang.F.e_var] of [get]. *)

  val copy : t -> t (** Duplicate the environment. Fresh chunks in the copy
                        are {i not} duplicated into the source environment. *)

  val join : t -> t -> Passive.t
  (** Make two environment pairwise equal {i via} the passive form.

      Missing chunks in one environment are added with the corresponding
      variable of the other environment. When both environments don't agree
      on a chunk, their variables are added to the passive form. *)

  val assigned : pre:t -> post:t -> domain -> pred Bag.t
  (** Make chunks equal outside of some domain.

      This is similar to [join], but outside the given footprint of an
      assigns clause. Although, the function returns the equality
      predicates instead of a passive form.

      Like in [join], missing chunks are reported from one side to the
      other one, and common chunks are added to the equality bag. *)

  val choose : t -> t -> t
  (** Make the union of each sigma, choosing the minimal variable
      in case of conflict.
      Both initial environments are kept unchanged. *)

  val merge : t -> t -> t * Passive.t * Passive.t
  (** Make the union of each sigma, choosing a {i new} variable for
      each conflict, and returns the corresponding joins.
      Both initial environments are kept unchanged. *)

  val merge_list : t list -> t * Passive.t list
  (** Same than {!merge} but for a list of sigmas. Much more efficient
      than folding merge step by step. *)

  val iter : (chunk -> var -> unit) -> t -> unit
  (** Iterates over the chunks and associated variables already
      accessed so far in the environment. *)

  val iter2 : (chunk -> var option -> var option -> unit) -> t -> t -> unit
  (** Same as [iter] for both environments. *)

  val havoc_chunk : t -> chunk -> t
  (** Generate a new fresh variable for the given chunk. *)

  val havoc : t -> domain -> t
  (** All the chunks in the provided footprint are generated and made fresh.

      Existing chunk variables {i outside} the footprint are copied into the new
      environment. The original environement itself is kept unchanged. More
      efficient than iterating [havoc_chunk] over the footprint.
  *)

  val havoc_any : call:bool -> t -> t
  (** All the chunks are made fresh. As an optimisation,
      when [~call:true] is set, only non-local chunks are made fresh.
      Local chunks are those for which [Chunk.is_frame] returns [true]. *)

  val remove_chunks : t -> domain -> t
  (** Return a copy of the environment where chunks in the footprint
      have been removed. Keep the original environment unchanged. *)

  val domain : t -> domain
  (** Footprint of a memory environment.
      That is, the set of accessed chunks so far in the environment. *)

  val union : domain -> domain -> domain (** Same as [Chunk.Set.union] *)
  val empty : domain (** Same as [Chunk.Set.empty] *)

  val writes : t sequence -> domain
  (** [writes s] indicates which chunks are new in [s.post] compared
      to [s.pre]. *)
end

(** Memory Models. *)
module type Model =
sig

  (** {2 Model Definition} *)

  val configure : WpContext.tuning
  (** Initializers to be run before using the model.
      Typically sets {!Context} values. *)

  val configure_ia: automaton -> vertex binder
  (** Given an automaton, return a vertex's binder.
      Currently used by the automata compiler to bind current vertex.
      See {!StmtSemantics}. *)

  val datatype : string
  (** For projectification. Must be unique among models. *)

  val hypotheses : unit -> MemoryContext.clause list
  (** Computes the memory model hypotheses including separation and validity
      clauses to be verified for this model. *)

  module Chunk : Chunk
  (** Memory model chunks. *)

  module Heap : Qed.Collection.S
    with type t = Chunk.t
  (** Chunks Sets and Maps. *)

  module Sigma : Sigma
    with type chunk = Chunk.t
     and module Chunk = Heap
  (** Model Environments. *)

  type loc
  (** Representation of the memory location in the model. *)

  type chunk = Chunk.t
  type sigma = Sigma.t
  type domain = Sigma.domain
  type segment = loc rloc

  (** {2 Reversing the Model} *)

  type state
  (** Internal (private) memory state description for later reversing the model. *)

  (** Returns a memory state description from a memory environement. *)
  val state : sigma -> state

  (** Try to interpret a term as an in-memory operation
      located at this program point. Only best-effort
      shall be performed, otherwise return [Mvalue].

      Recognized [Cil] patterns:
      - [Mvar x,[Mindex 0]] is rendered as [*x] when [x] has a pointer type
      - [Mmem p,[Mfield f;...]] is rendered as [p->f...] like in Cil
      - [Mmem p,[Mindex k;...]] is rendered as [p[k]...] to catch Cil [Mem(AddPI(p,k)),...] *)
  val lookup : state -> term -> mval

  (** Try to interpret a sequence of states into updates.

      The result shall be exhaustive with respect to values that are printed as [Sigs.mval]
      values at [post] label {i via} the [lookup] function.
      Otherwise, those values would not be pretty-printed to the user. *)
  val updates : state sequence -> Vars.t -> update Bag.t

  (** Propagate a sequent substitution inside the memory state. *)
  val apply : (term -> term) -> state -> state

  (** Debug *)
  val iter : (mval -> term -> unit) -> state -> unit

  val pretty : Format.formatter -> loc -> unit
  (** pretty printing of memory location *)

  (** {2 Memory Model API} *)

  val vars : loc -> Vars.t
  (** Return the logic variables from which the given location depend on. *)

  val occurs : var -> loc -> bool
  (** Test if a location depend on a given logic variable *)

  val null : loc
  (** Return the location of the null pointer *)

  val literal : eid:int -> Cstring.cst -> loc
  (** Return the memory location of a constant string,
      the id is a unique identifier. *)

  val cvar : varinfo -> loc
  (** Return the location of a C variable. *)

  val pointer_loc : term -> loc
  (** Interpret an address value (a pointer) as an abstract location.
      Might fail on memory models not supporting pointers. *)

  val pointer_val : loc -> term
  (** Return the adress value (a pointer) of an abstract location.
      Might fail on memory models not capable of representing pointers. *)

  val field : loc -> fieldinfo -> loc
  (** Return the memory location obtained by field access from a given
      memory location. *)

  val shift : loc -> c_object -> term -> loc
  (** Return the memory location obtained by array access at an index
      represented by the given {!term}. The element of the array are of
      the given {!c_object} type. *)

  val base_addr : loc -> loc
  (** Return the memory location of the base address of a given memory
      location. *)

  val base_offset : loc -> term
  (** Return the offset of the location, in bytes, from its base_addr. *)

  val block_length : sigma -> c_object -> loc -> term
  (**  Returns the length (in bytes) of the allocated block containing
       the given location. *)

  val cast : c_object sequence -> loc -> loc
  (** Cast a memory location into another memory location.
      For [cast ty loc] the cast is done from [ty.pre] to [ty.post].
      Might fail on memory models not supporting pointer casts. *)

  val loc_of_int : c_object -> term -> loc
  (** Cast a term representing an absolute memory address (to some c_object)
      given as an integer, into an abstract memory location. *)

  val int_of_loc : c_int -> loc -> term
  (** Cast a memory location into its absolute memory address,
      given as an integer with the given C-type. *)

  val domain : c_object -> loc -> domain
  (** Compute the set of chunks that hold the value of an object with
      the given C-type. It is safe to retun an over-approximation of the
      chunks involved. *)

  val load : sigma -> c_object -> loc -> loc value
  (** Return the value of the object of the given type at the given location in
      the given memory state. *)

  val copied : sigma sequence -> c_object -> loc -> loc -> equation list
  (**
     Return a set of equations that express a copy between two memory state.

     [copied sigma ty loc1 loc2] returns a set of formula expressing that the
     content for an object [ty] is the same in [sigma.pre] at [loc1] and in
     [sigma.post] at [loc2].
  *)

  val stored : sigma sequence -> c_object -> loc -> term -> equation list
  (**
     Return a set of formula that express a modification between two memory
     state.

     [copied sigma ty loc t] returns a set of formula expressing that
     [sigma.pre] and [sigma.post] are identical except for an object [ty] at
     location [loc] which is represented by [t] in [sigma.post].
  *)

  val assigned : sigma sequence -> c_object -> loc sloc -> equation list
  (**
     Return a set of formula that express that two memory state are the same
     except at the given set of memory location.

     This function can over-approximate the set of given memory location (e.g
     it can return [true] as if the all set of memory location was given).
  *)

  val is_null : loc -> pred
  (** Return the formula that check if a given location is null *)

  val loc_eq : loc -> loc -> pred
  val loc_lt : loc -> loc -> pred
  val loc_neq : loc -> loc -> pred
  val loc_leq : loc -> loc -> pred
  (** Memory location comparisons *)

  val loc_diff : c_object -> loc -> loc -> term
  (** Compute the length in bytes between two memory locations *)

  val valid : sigma -> acs -> segment -> pred
  (** Return the formula that tests if a memory state is valid
      (according to {!acs}) in the given memory state at the given
      segment.
  *)

  val frame : sigma -> pred list
  (** Assert the memory is a proper heap state preceeding the function
      entry point. *)

  val alloc : sigma -> varinfo list -> sigma
  (** Allocates new chunk for the validity of variables. *)

  val invalid : sigma -> segment -> pred
  (** Returns the formula that tests if the entire memory is invalid
      for write access. *)

  val scope : sigma sequence -> scope -> varinfo list -> pred list
  (** Manage the scope of variables.  Returns the updated memory model
      and hypotheses modeling the new validity-scope of the variables. *)

  val global : sigma -> term -> pred
  (** Given a pointer value [p], assumes this pointer [p] (when valid)
      is allocated outside the function frame under analysis. This means
      separated from the formals and locals of the function. *)

  val included : segment -> segment -> pred
  (** Return the formula that tests if two segment are included *)

  val separated : segment -> segment -> pred
  (** Return the formula that tests if two segment are separated *)

end

(* -------------------------------------------------------------------------- *)
(** {1 C and ACSL Compilers} *)
(* -------------------------------------------------------------------------- *)

(** Compiler for C expressions *)
module type CodeSemantics =
sig

  module M : Model (** The underlying memory model *)

  type loc = M.loc
  type nonrec value = loc value
  type nonrec result = loc result
  type sigma = M.Sigma.t

  val pp_value : Format.formatter -> value -> unit

  val cval : value -> term
  (** Evaluate an abstract value. May fail because of [M.pointer_val]. *)

  val cloc : value -> loc
  (** Interpret a value as a location. May fail because of [M.pointer_loc]. *)

  val cast : typ -> typ -> value -> value
  (** Applies a pointer cast or a conversion.

      [cast tr te ve] transforms a value [ve] with type [te] into a value
      with type [tr]. *)

  val equal_typ : typ -> value -> value -> pred
  (** Computes the value of [(a==b)] provided both [a] and [b] are values
      with the given type. *)

  val not_equal_typ : typ -> value -> value -> pred
  (** Computes the value of [(a==b)] provided both [a] and [b] are values
      with the given type. *)

  val equal_obj : c_object -> value -> value -> pred
  (** Same as [equal_typ] with an object type. *)

  val not_equal_obj : c_object -> value -> value -> pred
  (** Same as [not_equal_typ] with an object type. *)

  val exp : sigma -> exp -> value
  (** Evaluate the expression on the given memory state. *)

  val cond : sigma -> exp -> pred
  (** Evaluate the conditional expression on the given memory state. *)

  val lval : sigma -> lval -> loc
  (** Evaluate the left-value on the given memory state. *)

  val call : sigma -> exp -> loc
  (** Address of a function pointer.
      Handles [AddrOf], [StartOf] and [Lval] as usual. *)

  val instance_of : loc -> kernel_function -> pred
  (** Check whether a function pointer is (an instance of)
      some kernel function. Currently, the meaning
      of "{i being an instance of}" is simply equality. *)

  val loc_of_exp : sigma -> exp -> loc
  (** Compile an expression as a location.
      May (also) fail because of [M.pointer_val]. *)

  val val_of_exp : sigma -> exp -> term
  (** Compile an expression as a term.
      May (also) fail because of [M.pointer_loc]. *)

  val result : sigma -> typ -> result -> term
  (** Value of an abstract result container. *)

  val return : sigma -> typ -> exp -> term
  (** Return an expression with a given type.
      Short cut for compiling the expression, cast into the desired type,
      and finally converted to a term. *)

  val is_zero : sigma -> c_object -> loc -> pred
  (** Express that the object (of specified type) at the given location
      is filled with zeroes. *)

  (**
     Express that all objects in a range of locations have a given value.

     More precisely, [is_exp_range sigma loc ty a b v] express that
     value at [( ty* )loc + k] equals [v], forall [a <= k < b].
     Value [v=None] stands for zero.
  *)
  val is_exp_range :
    sigma -> loc -> c_object -> term -> term ->
    value option ->
    pred

  val unchanged : M.sigma -> M.sigma -> varinfo -> pred
  (** Express that a given variable has the same value in two memory states. *)

  type warned_hyp = Warning.Set.t * pred

  val init : sigma:M.sigma -> varinfo -> init option -> warned_hyp list
  (** Express that some variable has some initial value at the
      given memory state.

      Remark: [None] initializer are interpreted as zeroes. This is consistent
      with the [init option] associated with global variables in CIL,
      for which the default initializer are zeroes. There is no
      [init option] value associated with local initializers.
  *)

end

(** Compiler for ACSL expressions *)
module type LogicSemantics =
sig

  module M : Model (** Underlying memory model *)

  type loc = M.loc
  type nonrec value  = M.loc value
  type nonrec logic  = M.loc logic
  type nonrec region = M.loc region
  type nonrec result = M.loc result
  type sigma = M.Sigma.t

  (** {2 Frames}

      Frames are compilation environment for ACSL. A frame typically
      manages the current function, formal paramters, the memory environments
      at different labels and the [\result] and [\exit_status] values.

      The frame also holds the {i gamma} environment responsible for
      accumulating typing constraints, and the {i pool} for generating
      fresh logic variables.

      Notice that a [frame] is not responsible for holding the environment
      at label [Here], since this is managed by a specific compilation
      environment, see {!env} below.
  *)

  type frame
  val pp_frame : Format.formatter -> frame -> unit

  (** Get the current frame, or raise a fatal error if none. *)
  val get_frame : unit -> frame

  (** Execute the given closure with the specified current frame.
      The [Lang.gamma] and [Lang.pool] contexts are also set accordingly. *)
  val in_frame : frame -> ('a -> 'b) -> 'a -> 'b

  (** Get the memory environment at the given label.
      A fresh environment is created lazily if required.
      The label must {i not} be [Here]. *)
  val mem_at_frame : frame -> Clabels.c_label -> sigma

  (** Update a frame with a specific environment for the given label. *)
  val set_at_frame : frame -> Clabels.c_label -> sigma -> unit

  (** Same as [mem_at_frame] but for the current frame. *)
  val mem_frame : Clabels.c_label -> sigma

  (** Full featured constructor for frames, with fresh pool and gamma. *)
  val mk_frame :
    ?kf:Cil_types.kernel_function ->
    ?result:result ->
    ?status:Lang.F.var ->
    ?formals:value Cil_datatype.Varinfo.Map.t ->
    ?labels:sigma Clabels.LabelMap.t ->
    ?descr:string ->
    unit -> frame

  (** Make a local frame reusing the {i current} pool and gamma. *)
  val local : descr:string -> frame

  (** Make a fresh frame with the given function. *)
  val frame : kernel_function -> frame

  type call (** Internal call data. *)

  (** Create call data from the callee point of view,
      deriving data (gamma and pools) from the current frame.
      If [result] is specified, the called function will stored its result
      at the provided location in the current frame (the callee). *)
  val call : ?result:M.loc -> kernel_function -> value list -> call

  (** Derive a frame from the call data suitable for compiling the
      called function contracts in the provided pre-state. *)
  val call_pre   : sigma -> call -> sigma -> frame

  (** Derive a frame from the call data suitable for compiling the
      called function contracts in the provided pre-state and post-state. *)
  val call_post  : sigma -> call -> sigma sequence -> frame

  (** Result type of the current function in the current frame. *)
  val return : unit -> typ

  (** Result location of the current function in the current frame. *)
  val result : unit -> result

  (** Exit status for the current frame. *)
  val status : unit -> var

  (** Returns the current gamma environment from the current frame. *)
  val guards : frame -> pred list

  (** {2 Compilation Environment} *)

  type env
  (**
     Compilation environment for terms and predicates. Manages
     the {i current} memory state and the memory state at [Here].

     Remark: don't confuse the {i current} memory state with the
     memory state {i at label} [Here]. The current memory state is the one
     we have at hand when compiling a term or a predicate. Hence, inside
     [\at(e,L)] the current memory state when compiling [e] is the one at [L].
  *)

  (** Create a new environment.

      Current and [Here] memory points are initialized to [~here], if
      provided.

      The logic variables stand for
      formal parameters of ACSL logic function and ACSL predicates. *)
  val mk_env :
    ?here:sigma ->
    ?lvars:logic_var list ->
    unit -> env

  (** The {i current} memory state. Must be propertly initialized
      with a specific {!move} before. *)
  val current : env -> sigma

  (** Move the compilation environment to the specified [Here] memory state.
      This memory state becomes also the new {i current} one. *)
  val move_at : env -> sigma -> env

  (** Returns the memory state at the requested label.
      Uses the local environment for [Here] and the current frame
      otherwize. *)
  val mem_at : env -> Clabels.c_label -> sigma

  (** Returns a new environment where the current memory state is
      moved to to the corresponding label. Suitable for compiling [e] inside
      [\at(e,L)] ACSL construct. *)
  val env_at : env -> Clabels.c_label -> env

  (** {2 Compilers} *)

  (** Compile a term l-value into a (typed) abstract location *)
  val lval : env -> Cil_types.term_lval -> Cil_types.typ * M.loc

  (** Compile a term expression. *)
  val term : env -> Cil_types.term -> term

  (** Compile a predicate. The polarity is used to generate a weaker or
      stronger predicate in case of unsupported feature from WP or the
      underlying memory model. *)
  val pred : polarity -> env -> Cil_types.predicate -> pred

  (** Compile a term representing a set of memory locations into an abstract
      region. When [~unfold:true], compound memory locations are expanded
      field-by-field. *)
  val region : env -> unfold:bool -> Cil_types.term -> region

  (** Computes the region assigned by a list of froms. *)
  val assigned_of_froms :
    env -> unfold:bool -> from list -> region

  (** Computes the region assigned by an assigns clause.
      [None] means everyhting is assigned. *)
  val assigned_of_assigns :
    env -> unfold:bool -> assigns -> region option

  (** Same as [term] above but reject any set of locations. *)
  val val_of_term : env -> Cil_types.term -> term

  (** Same as [term] above but expects a single loc or a single
      pointer value. *)
  val loc_of_term : env -> Cil_types.term -> loc

  (** Compile a lemma definition. *)
  val lemma : LogicUsage.logic_lemma -> Definitions.dlemma

  (** {2 Regions} *)

  (** Qed variables appearing in a region expression. *)
  val vars : region -> Vars.t

  (** Member of vars. *)
  val occurs : var -> region -> bool

  (** Check assigns inclusion.
      Compute a formula that checks whether written locations are either
      invalid (at the given memory location)
      or included in some assignable region. *)
  val check_assigns : sigma -> written:region -> assignable:region -> pred

end

(** Compiler for Performing Assigns *)
module type LogicAssigns = sig

  module M : Model
  module L : LogicSemantics with module M = M
  open M

  (** Memory footprint of a region. *)
  val domain : loc region -> Heap.set

  (** Relates two memory states corresponding to an assigns clause
      with the specified set of locations. *)
  val apply_assigns : sigma sequence -> loc region -> pred list

end

(** All Compilers Together *)
module type Compiler = sig
  module M : Model
  module C : CodeSemantics with module M = M
  module L : LogicSemantics with module M = M
  module A : LogicAssigns with module M = M and module L = L
end
end
module Mstate : sig
# 1 "./Mstate.mli"
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

open Lang.F
open Sigs

(* -------------------------------------------------------------------------- *)
(* --- L-Val Utility                                                      --- *)
(* -------------------------------------------------------------------------- *)

val index : s_lval -> term -> s_lval
val field : s_lval -> Cil_types.fieldinfo -> s_lval
val equal : s_lval -> s_lval -> bool

(* -------------------------------------------------------------------------- *)
(* --- Memory State Pretty Printing Information                           --- *)
(* -------------------------------------------------------------------------- *)

type 'a model
type state

val create : (module Model with type Sigma.t = 'a) -> 'a model
val state : 'a model -> 'a -> state

val lookup : state -> term -> mval
val apply : (term -> term) -> state -> state
val iter : (mval -> term -> unit) -> state -> unit
val updates : state sequence -> Vars.t -> update Bag.t
end
module Conditions : sig
# 1 "./Conditions.mli"
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

(* -------------------------------------------------------------------------- *)
(* --- Weakest Pre Accumulator                                            --- *)
(* -------------------------------------------------------------------------- *)

open Cil_types
open Lang
open Lang.F

(** Predicates *)
val forall_intro: Lang.F.pred -> Lang.F.pred list * Lang.F.pred
val exist_intro: Lang.F.pred -> Lang.F.pred

(** Sequent *)

type step = private {
  mutable id : int ; (** See [index] *)
  size : int ;
  vars : Vars.t ;
  stmt : stmt option ;
  descr : string option ;
  deps : Property.t list ;
  warn : Warning.Set.t ;
  condition : condition ;
}

and condition =
  | Type of pred
  | Have of pred
  | When of pred
  | Core of pred
  | Init of pred
  | Branch of pred * sequence * sequence
  | Either of sequence list
  | State of Mstate.state

and sequence (** List of steps *)

type sequent = sequence * F.pred

val pretty : (Format.formatter -> sequent -> unit) ref

val step :
  ?descr:string ->
  ?stmt:stmt ->
  ?deps:Property.t list ->
  ?warn:Warning.Set.t ->
  condition -> step

(** Updates the condition of a step and merges [descr], [deps] and [warn] *)
val update_cond :
  ?descr:string ->
  ?deps:Property.t list ->
  ?warn:Warning.Set.t ->
  step ->
  condition -> step

val is_true : sequence -> bool (** Only true or empty steps *)
val is_empty : sequence -> bool (** No step at all *)
val vars_hyp : sequence -> Vars.t
val vars_seq : sequent -> Vars.t

val empty : sequence
val trivial : sequent
val sequence : step list -> sequence
val seq_branch : ?stmt:stmt -> F.pred -> sequence -> sequence -> sequence

val append : sequence -> sequence -> sequence
val concat : sequence list -> sequence

(** Iterate only over the head steps of the sequence *)
val iter : (step -> unit) -> sequence -> unit

(** The internal list of steps *)
val list : sequence -> step list

val size : sequence -> int

val steps : sequence -> int
(** Attributes unique indices to every [step.id] in the sequence, starting from zero.
    Returns the number of steps in the sequence. *)

val index : sequent -> unit
(** Compute steps' id of sequent left hand-side.
    Same as [ignore (steps (fst s))]. *)

val step_at : sequence -> int -> step
(** Retrieve a step by [id] in the sequence.
    The [index] function {i must} have been called on the sequence before
    retrieving the index properly.
    @raise Not_found if the index is out of bounds. *)

val is_trivial : sequent -> bool

(** {2 Transformations} *)

val map_condition : (pred -> pred) -> condition -> condition
val map_step : (pred -> pred) -> step -> step
val map_sequence : (pred -> pred) -> sequence -> sequence
val map_sequent : (pred -> pred) -> sequent -> sequent

val insert : ?at:int -> step -> sequent -> sequent
(** Insert a step in the sequent immediately [at] the specified position.
    Parameter [at] can be [size] to insert at the end of the sequent (default).
    @raise Invalid_argument if the index is out of bounds. *)

val replace : at:int -> step -> sequent -> sequent
(** replace a step in the sequent, the one [at] the specified position.
    @raise Invalid_argument if the index is out of bounds. *)

val subst : (term -> term) -> sequent -> sequent
(** Apply the atomic substitution recursively using [Lang.F.p_subst f].
    Function [f] should only transform the head of the predicate, and can assume
    its sub-terms have been already substituted. The atomic substitution is also applied
    to predicates.
    [f] should raise [Not_found] on terms that must not be replaced
*)

val introduction : sequent -> sequent option
(** Performs existential, universal and hypotheses introductions *)

val introduction_eq : sequent -> sequent
(** Same as [introduction] but returns the same sequent is None *)

val lemma : pred -> sequent
(** Performs existential, universal and hypotheses introductions *)

val head : step -> pred (** Predicate for Have and such, Condition for Branch, True for Either *)
val have : step -> pred (** Predicate for Have and such, True for any other *)

val condition : sequence -> pred (** With free variables kept. *)
val close : sequent -> pred (** With free variables {i quantified}. *)

val at_closure : (sequent -> sequent ) -> unit (** register a transformation applied just before close *)

(** {2 Bundles}

    Bundles are {i mergeable} pre-sequences.
    This the key structure for merging hypotheses with linear complexity
    during backward weakest pre-condition calculus.
*)

type bundle

type 'a attributed =
  ( ?descr:string ->
    ?stmt:stmt ->
    ?deps:Property.t list ->
    ?warn:Warning.Set.t ->
    'a )

val nil : bundle
val occurs : F.var -> bundle -> bool
val intersect : F.pred -> bundle -> bool
val merge : bundle list -> bundle
val domain : F.pred list -> bundle -> bundle
val intros : F.pred list -> bundle -> bundle
val state : ?descr:string -> ?stmt:stmt -> Mstate.state -> bundle -> bundle
val assume : (?init:bool -> F.pred -> bundle -> bundle) attributed
val assume_when : (?init:bool -> F.pred -> bundle -> bundle) attributed
val branch : (F.pred -> bundle -> bundle -> bundle) attributed
val either : (bundle list -> bundle) attributed
val extract : bundle -> F.pred list
val extract_when : bundle -> F.pred option list  * F.pred option list 
val bundle : bundle -> sequence

(** {2 Simplifier} *)

val clean : sequent -> sequent
val filter : sequent -> sequent
val parasite : sequent -> sequent
val simplify : ?solvers:simplifier list -> ?intros:int -> sequent -> sequent
val pruning : ?solvers:simplifier list -> sequent -> sequent

(* -------------------------------------------------------------------------- *)
end
module Filtering : sig
# 1 "./Filtering.mli"
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

(* -------------------------------------------------------------------------- *)
(** Sequent Cleaning *)
(* -------------------------------------------------------------------------- *)

open Lang

(**
   Erase parts of a predicate that do not satisfies the condition.
   The erased parts are replaced by:
   - [true] when [~polarity:false] (for hypotheses)
   - [false] when [~polarity:true] (for goals)

   Hence, we have:
   - [filter ~polarity:true f p ==> p]
   - [p ==> filter ~polarity:false f p]

   See [theory/filtering.why] for proofs.
*)

val filter : polarity:bool -> (F.pred -> bool) -> F.pred -> F.pred

open Conditions

val compute : ?anti:bool -> sequent -> sequent




(* -------------------------------------------------------------------------- *)
end
module Plang : sig
# 1 "./Plang.mli"
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

open Lang
open Lang.F

(** Lang Pretty-Printer *)

type scope = Qed.Engine.scope
module Env : Qed.Engine.Env with type term := term

type pool
val pool : unit -> pool
val alloc_e : pool -> (var -> unit) -> term -> unit
val alloc_p : pool -> (var -> unit) -> pred -> unit
val alloc_xs : pool -> (var -> unit) -> Vars.t -> unit
val alloc_domain : pool -> Vars.t
val sanitizer : string -> string

type iformat = [ `Hex | `Dec | `Bin ]
type rformat = [ `Ratio | `Float | `Double ]

class engine :
  object
    inherit [Z.t,ADT.t,Field.t,Fun.t,tau,var,term,Env.t] Qed.Engine.engine
    method get_iformat : iformat
    method set_iformat : iformat -> unit
    method get_rformat : rformat
    method set_rformat : rformat -> unit
    method marks : Env.t * Lang.F.marks
    method pp_pred : Format.formatter -> pred -> unit
    method lookup : term -> scope
    (**/**)
    inherit Lang.idprinting
    method sanitize : string -> string
    method infoprover : 'a. 'a Lang.infoprover -> 'a
    method op_spaced : string -> bool
    (**/**)
  end
end
module Pcfg : sig
# 1 "./Pcfg.mli"
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

open Sigs
open Lang
open Lang.F

type env
type label

type value =
  | Term
  | Addr of s_lval
  | Lval of s_lval * label
  | Chunk of string * label

val create : unit -> env
val register : Conditions.sequence -> env

val at : env -> id:int -> label
val find : env -> F.term -> value
val updates : env -> label Sigs.sequence -> Vars.t -> Sigs.update Bag.t
val visible : label -> bool
val subterms : env -> (F.term -> unit) -> F.term -> bool
val prev : label -> label list
val next : label -> label list
val iter : (Sigs.mval -> term -> unit) -> label -> unit
val branching : label -> bool

class virtual engine :
  object
    method virtual pp_atom : Format.formatter -> term -> unit
    method virtual pp_flow : Format.formatter -> term -> unit

    method is_atomic_lv : s_lval -> bool

    method pp_ofs : Format.formatter -> s_offset -> unit
    method pp_offset : Format.formatter -> s_offset list -> unit
    method pp_host : Format.formatter -> s_host -> unit (** current state *)
    method pp_lval : Format.formatter -> s_lval -> unit (** current state *)
    method pp_addr : Format.formatter -> s_lval -> unit
    method pp_label : Format.formatter -> label -> unit (** label name *)
    method pp_chunk : Format.formatter -> string -> unit (** chunk name *)
  end
end
module Pcond : sig
# 1 "./Pcond.mli"
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

open Qed.Plib
open Conditions

(** {2 All-in-one printers} *)

val dump : bundle printer
val bundle : ?clause:string -> bundle printer
val sequence : ?clause:string -> sequence printer
val pretty : sequent printer

(** {2 Low-level API} *)

open Lang.F
type env = Plang.Env.t

val alloc_hyp : Plang.pool -> (var -> unit) -> sequence -> unit
val alloc_seq : Plang.pool -> (var -> unit) -> sequent -> unit

(** Sequent Printer Engine. Uses the following [CSS]:
    - ["wp:clause"] for all clause keywords
    - ["wp:comment"] for descriptions
    - ["wp:warning"] for warnings
    - ["wp:property"] for properties
*)

class engine : #Plang.engine ->
  object

    (** {2 Printer Components} *)
    method name : env -> term -> string (** Generate a name for marked term *)
    method mark : marks -> step -> unit (** Marks terms to share in step *)
    method pp_clause : string printer (** Default: ["@{<wp:clause>...}"] *)
    method pp_stmt : string printer (** Default: ["@{<wp:stmt>...}"] *)
    method pp_comment : string printer (** Default: ["@{<wp:comment>(* ... *)}"] *)
    method pp_property : Property.t printer (** Default: ["@{<wp:property>(* ... *)}"] *)
    method pp_warning : Warning.t printer (** Default: ["@{<wp:warning>Warning}..."] *)
    method pp_name : string printer (** Default: [Format.pp_print_string] *)
    method pp_core : term printer (** Default: [plang#pp_sort] *)

    method pp_definition : Format.formatter -> string -> term -> unit
    method pp_intro : step:step -> clause:string -> ?dot:string -> pred printer
    method pp_condition : step:step -> condition printer
    method pp_block : clause:string -> sequence printer
    method pp_goal : pred printer

    method pp_step : step printer
    (** Assumes an "<hv>" box is opened. *)

    method pp_block : clause:string -> sequence printer
    (** Assumes an "<hv>" box is opened and all variables are named. *)

    method pp_sequence : clause:string -> sequence printer
    (** Assumes an "<hv>" box is opened {i and} all variables are declared.
        (recursively used) *)

    method pp_sequent : sequent printer
    (** Print the sequent in global environment. *)

    method pp_esequent : env -> sequent printer
    (** Print the sequent in the given environment.
        The environment is enriched with the shared terms. *)

  end

(* -------------------------------------------------------------------------- *)
(* --- State-Aware Printers                                               --- *)
(* -------------------------------------------------------------------------- *)

class state :
  object
    inherit Plang.engine
    inherit Pcfg.engine
    method clear : unit
    method set_sequence : Conditions.sequence -> unit
    method set_domain : Vars.t -> unit (** Default is sequence's domain *)
    method domain : Vars.t
    method label_at : id:int -> Pcfg.label
    method updates : Pcfg.label Sigs.sequence -> Sigs.update Bag.t
    method pp_at : Format.formatter -> Pcfg.label -> unit
    method pp_update : Pcfg.label -> Format.formatter -> Sigs.update -> unit
    method pp_value : Format.formatter -> term -> unit
  end

class sequence : #state ->
  object
    inherit engine
    method set_sequence : Conditions.sequence -> unit
    (** Initialize state with this sequence *)
    method set_goal : pred -> unit
    (** Adds goal to state domain *)
    method set_sequent : sequent -> unit
    (** Set sequence and goal *)
    method get_state : bool
    (** If [true], states are rendered when printing sequences. *)
    method set_state : bool -> unit
    (** If set to [false], states rendering is deactivated. *)
  end
end
module CodeSemantics : sig
# 1 "./CodeSemantics.mli"
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

(* -------------------------------------------------------------------------- *)
(* --- C-Code Translation                                                 --- *)
(* -------------------------------------------------------------------------- *)


module Make(M : Sigs.Model) : Sigs.CodeSemantics with module M = M
end
module LogicCompiler : sig
# 1 "./LogicCompiler.mli"
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

(* -------------------------------------------------------------------------- *)
(* --- Compilation of ACSL Logic-Info                                     --- *)
(* -------------------------------------------------------------------------- *)

open LogicUsage
open Cil_types
open Cil_datatype
open Clabels
open Lang
open Lang.F
open Sigs
open Definitions

type polarity = [ `Positive | `Negative | `NoPolarity ]

module Make( M : Sigs.Model ) :
sig

  (** {3 Definitions} *)

  type value = M.loc Sigs.value
  type logic = M.loc Sigs.logic
  type result = M.loc Sigs.result
  type sigma = M.Sigma.t
  type chunk = M.Chunk.t

  (** {3 Frames} *)

  type call
  type frame

  val pp_frame : Format.formatter -> frame -> unit

  val local : descr:string -> frame
  val frame : kernel_function -> frame
  val call : ?result:M.loc -> kernel_function -> value list -> call
  val call_pre   : sigma -> call -> sigma -> frame
  val call_post  : sigma -> call -> sigma sequence -> frame

  val mk_frame :
    ?kf:Cil_types.kernel_function ->
    ?result:result ->
    ?status:Lang.F.var ->
    ?formals:value Varinfo.Map.t ->
    ?labels:sigma Clabels.LabelMap.t ->
    ?descr:string ->
    unit -> frame

  val formal : varinfo -> value option
  val return : unit -> typ
  val result : unit -> result
  val status : unit -> var
  val trigger : trigger -> unit

  val guards : frame -> pred list
  val mem_frame : c_label -> sigma
  val mem_at_frame : frame -> c_label -> sigma
  val set_at_frame : frame -> c_label -> sigma -> unit

  val in_frame : frame -> ('a -> 'b) -> 'a -> 'b
  val get_frame : unit -> frame

  (** {3 Environment} *)

  type env

  val mk_env : ?here:sigma -> ?lvars:Logic_var.t list -> unit -> env
  val current : env -> sigma
  val move_at : env -> sigma -> env
  val env_at : env -> c_label -> env
  val mem_at : env -> c_label -> sigma
  val env_let : env -> logic_var -> logic -> env
  val env_letp : env -> logic_var -> pred -> env
  val env_letval : env -> logic_var -> value -> env

  (** {3 Compiler} *)

  val term : env -> Cil_types.term -> term
  val pred : polarity -> env -> predicate -> pred
  val logic : env -> Cil_types.term -> logic
  val region : env -> unfold:bool -> Cil_types.term -> M.loc Sigs.region
  (** When [~unfold:true], decompose compound regions field by field *)

  val bootstrap_term : (env -> Cil_types.term -> term) -> unit
  val bootstrap_pred : (polarity -> env -> predicate -> pred) -> unit
  val bootstrap_logic : (env -> Cil_types.term -> logic) -> unit
  val bootstrap_region :
    (env -> unfold:bool -> Cil_types.term -> M.loc Sigs.region) -> unit

  (** {3 Application} *)

  val call_fun : env -> logic_info
    -> logic_label list
    -> F.term list -> F.term

  val call_pred : env -> logic_info
    -> logic_label list
    -> F.term list -> F.pred

  (** {3 Logic Variable and ACSL Constants} *)

  val logic_var : env -> logic_var -> logic
  val logic_info : env -> logic_info -> pred option

  (** {3 Logic Lemmas} *)

  val lemma : logic_lemma -> dlemma

end
end
module LogicSemantics : sig
# 1 "./LogicSemantics.mli"
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

(* -------------------------------------------------------------------------- *)
(* --- ACSL Translation                                                   --- *)
(* -------------------------------------------------------------------------- *)

module Make(M : Sigs.Model) : Sigs.LogicSemantics with module M = M
end
module Sigma : sig
# 1 "./Sigma.mli"
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

(* -------------------------------------------------------------------------- *)
(* --- Generic Sigma Factory                                              --- *)
(* -------------------------------------------------------------------------- *)

module Make
    (C : Sigs.Chunk)
    (H : Qed.Collection.S with type t = C.t) :
  Sigs.Sigma with type chunk = C.t
              and module Chunk = H
end
module MemVar : sig
# 1 "./MemVar.mli"
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

(* -------------------------------------------------------------------------- *)
(* --- No-Aliasing Memory Model                                           --- *)
(* -------------------------------------------------------------------------- *)

open Cil_types

module type VarUsage =
sig
  val datatype : string
  val param : varinfo -> MemoryContext.param
  (** Memory Model Hypotheses *)
  val hypotheses : unit -> MemoryContext.clause list
end

module Make(V : VarUsage)(M : Sigs.Model) : Sigs.Model
end
module MemTyped : sig
# 1 "./MemTyped.mli"
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

(* -------------------------------------------------------------------------- *)
(* --- Typed Memory Model                                                 --- *)
(* -------------------------------------------------------------------------- *)

include Sigs.Model

type pointer = NoCast | Fits | Unsafe
val pointer : pointer Context.value
end
module CfgCompiler : sig
# 1 "./CfgCompiler.mli"
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

open Sigs
open Cil_types
open Lang

(** {2 Control Flow Graphs}

    The semantics of a {i cfg} is a collection of execution traces.  We
    introduce the notion of {i node} which represent a program point.
    In case of loop unrolling of function inlining, a node
    generalize the notion of [stmt] : two distinct nodes may refer to the same
    instruction at different memory states.

    We introduce an interpretation I as a partial mapping from nodes [n:node] to
    memory states [s:M.sigma], denoted I(n). The notation I(n) seen as a predicate
    indicates if `n` is in the partial mapping.

    Given a cfg, a node can be associated to {i assumptions} to filter
    interpretation against the memory state at this point.

    Effects and predicates are defined {i wrt} some fresh memory states, and can
    be duplicated at different nodes, each instance being mapped to different
    memory states.

*)

type mode = [
  | `Tree
  | `Bool_Backward
  | `Bool_Forward
]

module type Cfg =
sig

  (** The memory model used. *)
  module S : Sigma

  (** Program point along a trace. *)
  module Node : sig
    type t
    module Map : Qed.Idxmap.S with type key = t
    module Set : Qed.Idxset.S with type elt = t
    module Hashtbl : Hashtbl.S with type key = t
    val pp: Format.formatter -> t -> unit
    val create: unit -> t
    val equal: t -> t -> bool
  end

  type node = Node.t

  (** fresh node *)
  val node : unit -> node

  (** {2 Relocatable Formulae}
      Can be created once with fresh environment, and used several
      times on different memory states. *)

  (** Relocatable condition *)
  module C :
  sig
    type t

    val equal : t -> t -> bool

    (** Bundle an equation with the sigma sequence that created it. *)
    val create : S.t -> F.pred -> t
    val get : t -> F.pred
    val reads : t -> S.domain
    val relocate : S.t -> t -> t
  end

  (** Relocatable predicate *)
  module P :
  sig
    type t
    val pretty : Format.formatter -> t -> unit

    (** Bundle an equation with the sigma sequence that created it.
        [| create m p |] = [| p |]
    *)
    val create : S.t Node.Map.t -> F.pred -> t
    val get: t -> F.pred
    val reads : t -> S.domain Node.Map.t
    val nodes : t -> Node.Set.t
    val relocate : S.t Node.Map.t -> t -> t
    (** [| relocate m' (create m p) |] = [| p{ } |] *)

    val to_condition: t -> (C.t * Node.t option) option
  end

  (** Relocatable term *)
  module T :
  sig
    type t
    val pretty : Format.formatter -> t -> unit

    (** Bundle a term with the sigma sequence that created it. *)
    val create : S.t Node.Map.t -> F.term -> t
    val get: t -> F.term
    val reads : t -> S.domain Node.Map.t
    val relocate : S.t Node.Map.t -> t -> t
    val init  : Node.Set.t -> (S.t Node.Map.t -> F.term) -> t
    val init' : Node.t -> (S.t -> F.term) -> t
  end


  (** Relocatable effect (a predicate that depend on two states). *)
  module E : sig
    type t
    val pretty: Format.formatter -> t -> unit

    (** Bundle an equation with the sigma sequence that created it *)
    val create : S.t sequence -> F.pred -> t
    val get : t -> F.pred
    val reads : t -> S.domain
    val writes : t -> S.domain
    (** as defined by S.writes *)
    val relocate : S.t sequence -> t -> t
  end

  type cfg (** Structured collection of traces. *)

  val dump_env: name:string -> cfg -> unit
  val output_dot: out_channel -> ?checks:P.t Bag.t -> cfg -> unit

  val nop : cfg
  (** Structurally, [nop] is an empty execution trace.
      Hence, [nop] actually denotes all possible execution traces.
      This is the neutral element of [concat].

      Formally: all interpretations I verify nop: [| nop |]_I
  *)

  val add_tmpnode: node -> cfg
  (** Set a node as temporary. Information about its path predicate or
      sigma can be discarded during compilation *)

  val concat : cfg -> cfg -> cfg
  (** The concatenation is the intersection of all
      possible collection of traces from each cfg.

      [concat] is associative, commutative,
      has [nop] as neutral element.

      Formally: [| concat g1 g2 |]_I iff [| g1 |]_I and [| g2 |]_I
  *)

  val meta : ?stmt:stmt -> ?descr:string -> node -> cfg
  (** Attach meta informations to a node.
      Formally, it is equivalent to [nop]. *)

  val goto : node -> node -> cfg
  (** Represents all execution traces [T] such that, if [T] contains node [a],
      [T] also contains node [b] and memory states at [a] and [b] are equal.

      Formally: [| goto a b |]_I iff (I(a) iff I(b))
  *)

  val branch : node -> C.t -> node -> node -> cfg
  (** Structurally corresponds to an if-then-else control-flow.
      The predicate [P] shall reads only memory state at label [Here].

      Formally: [| branch n P a b |]_I iff (   (I(n) iff (I(a) \/ I(b)))
                                            /\ (I(n) implies (if P(I(n)) then I(a) else I(b)))  )
  *)

  val guard : node -> C.t -> node -> cfg
  (** Structurally corresponds to an assume control-flow.
      The predicate [P] shall reads only memory state at label [Here].

      Formally: [| guard n P a |]_I iff (   (I(n) iff I(a))
                                            /\ (I(n) implies [| P |]_I  ) )
  *)

  val guard' : node -> C.t -> node -> cfg
  (** Same than guard but the condition is negated *)

  val either : node -> node list -> cfg
  (** Structurally corresponds to an arbitrary choice among the different
      possible executions.

      [either] is associative and commutative. [either a []] is
      very special, since it denotes a cfg with {i no} trace. Technically,
      it is equivalent to attaching an [assert \false] annotation to node [a].

      Formally: [| either n [a_1;...;a_n] } |]_I iff ( I(n) iff (I(a_1) \/ ... I(a_n)))
  *)


  val implies : node -> (C.t * node) list -> cfg
  (**
     implies is the dual of either. Instead of being a non-deterministic choice,
     it takes the choices that verify its predicate.

      Formally: [| either n [P_1,a_1;...;P_n,a_n] } |]_I iff ( I(n) iff (I(a_1) \/ ... I(a_n))
                                                              /\  I(n) implies [| P_k |]_I implies I(a_k)
  *)


  val effect : node -> E.t -> node -> cfg
  (** Represents all execution trace [T] such that, if [T] contains node [a],
      then [T] also contains [b] with the given effect on corresponding
      memory states.

      Formally: [| effect a e b |]_I iff (( I(a) iff I(b) ) /\ [| e |]_I )
  *)

  val assume : P.t -> cfg
  (** Represents execution traces [T] such that, if [T] contains
      every node points in the label-map, then the condition holds over the
      corresponding memory states. If the node-map is empty,
      the condition must hold over all possible execution path.

      Formally: [| assume P |]_I iff [| P |]_I
  *)

  val havoc : node -> effects:node sequence -> node -> cfg
  (** Inserts an assigns effect between nodes [a] and [b], correspondings
      to all the written memory chunks accessible in execution paths delimited
      by the [effects] sequence of nodes.

      Formally: [| havoc a s b |]_I is verified if there is no path between s.pre and s.path,
      otherwise if (I(a) iff I(b) and if I(a) is defined then I(a) and I(b) are equal
      for all the chunks that are not in the written domain of an effect that can be found
      between [s.pre] to [s.post].

      Note: the effects are collected in the {i final} control-flow,
      when {!compile} is invoked. The portion of the sub-graph in the sequence
      shall be concatenated to the [cfg] before compiling-it, otherwize it would be
      considered empty and [havoc] would be a nop (no connection between a and b).
  *)

  (** {2 Path-Predicates}

      The compilation of cfg control-flow into path predicate
      is performed by allocating fresh environments with optimized variable
      allocation. Only the relevant path between the nodes
      is extracted. Other paths in the cfg are pruned out.
  *)

  (** Extract the nodes that are between the start node and the final
      nodes and returns how to observe a collection of states indexed
      by nodes. The returned maps gives, for each reachable node, a
      predicate representing paths that reach the node and the memory
      state at this node.

      Nodes absent from the map are unreachable. Whenever possible,
      predicate [F.ptrue] is returned for inconditionally accessible
      nodes.

      ~name: identifier used for debugging

  *)

  val compile : ?name:string -> ?mode:mode -> node -> Node.Set.t -> S.domain Node.Map.t ->
    cfg -> F.pred Node.Map.t * S.t Node.Map.t * Conditions.sequence

end

module Cfg(S:Sigma) : Cfg with module S = S
end
module StmtSemantics : sig
# 1 "./StmtSemantics.mli"
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

open Cil_types
open Clabels

module Make(Compiler : Sigs.Compiler) :
sig

  module Cfg : CfgCompiler.Cfg with module S = Compiler.M.Sigma

  type node = Cfg.node
  type goal = {
    goal_pred : Cfg.P.t;
    goal_prop : WpPropId.prop_id;
  }
  type cfg = Cfg.cfg
  type paths = {
    paths_cfg : cfg;
    paths_goals : goal Bag.t;
  }

  val goals_nodes: goal Bag.t -> Cfg.Node.Set.t

  exception LabelNotFound of c_label

  (** Compilation environment *)

  type env

  val empty_env : Kernel_function.t -> env
  val bind : c_label -> node -> env -> env

  val result : env -> Lang.F.var

  val (@^) : paths -> paths -> paths (** Same as [Cfg.concat] *)
  val (@*) : env -> ( c_label * node ) list -> env (** fold bind *)
  val (@:) : env -> c_label -> node
  (** LabelMap.find with refined excpetion.
      @raise LabelNotFound instead of [Not_found] *)

  val (@-) : env -> (c_label -> bool) -> env

  val sequence : (env -> 'a -> paths) -> env -> 'a list -> paths
  (** Chain compiler by introducing fresh nodes between each element
      of the list. For each consecutive [x;y] elements, a fresh node [n]
      is created, and [x] is compiled with [Next:n] and [y] is compiled with
      [Here:n]. *)

  val choice : ?pre:c_label -> ?post:c_label ->
    (env -> 'a -> paths) -> env -> 'a list -> paths
  (** Chain compiler in parallel, between labels [~pre] and [~post], which
      defaults to resp. [here] and [next].
      The list of eventualities is exhastive, hence an [either] assumption
      is also inserted. *)

  val parallel : ?pre:c_label -> ?post:c_label ->
    (env -> 'a -> Cfg.C.t * paths) -> env -> 'a list -> paths
  (** Chain compiler in parallel, between labels [~pre] and [~post], which
      defaults to resp. [here] and [next].
      The list of eventualities is exhastive, hence an [either] assumption
      is also inserted. *)

  (** {2 Instructions Compilation}

      Each instruction or statement is typically compiled between
      [Here] and [Next] nodes in the [flow]. [Pre], [Post] and [Exit] are
      reserved for the entry and exit points of current function.
      in [flow] are used when needed such as [Break] and [Continue] and
      should be added before calling.
  *)

  val set : env -> lval -> exp -> paths
  val scope : env -> Sigs.scope -> varinfo list -> paths
  val instr : env -> instr -> paths
  val return : env -> exp option -> paths
  val assume : Cfg.P.t -> paths

  val call_kf : env -> lval option -> kernel_function -> exp list -> paths
  val call : env -> lval option -> exp -> exp list -> paths

  (** {2 ACSL Compilation}  *)

  val spec : env -> spec -> paths
  val assume_ : env -> Sigs.polarity -> predicate -> paths
  val assigns : env -> assigns -> paths
  val froms : env -> from list -> paths

  (** {2 Automata Compilation} *)

  val automaton : env -> Interpreted_automata.automaton -> paths

  val init: is_pre_main:bool -> env -> paths

  (** {2 Full Compilation}

      Returns the set of all paths for the function, with all proof
      obligations. The returned node corresponds to the [Init] label. *)
  val compute_kf: Kernel_function.t -> paths * node
end
end
module Factory : sig
# 1 "./Factory.mli"
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

(* -------------------------------------------------------------------------- *)
(* --- Model Factory                                                      --- *)
(* -------------------------------------------------------------------------- *)

type mheap = Hoare | ZeroAlias | Region | Typed of MemTyped.pointer
type mvar = Raw | Var | Ref | Caveat

type setup = {
  mvar : mvar ;
  mheap : mheap ;
  cint : Cint.model ;
  cfloat : Cfloat.model ;
}

type driver = LogicBuiltins.driver

val ident : setup -> string
val descr : setup -> string
val compiler : mheap -> mvar -> (module Sigs.Compiler)
val configure : setup -> driver -> WpContext.tuning
val instance : setup -> driver -> WpContext.model
val default : setup (** ["Var,Typed,Nat,Real"] memory model. *)
val parse :
  ?default:setup ->
  ?warning:(string -> unit) ->
  string list -> setup
(**
   Apply specifications to default setup.
   Default setup is [Factory.default].
   Default warning is [Wp_parameters.abort]. *)
end
module Driver : sig
# 1 "./driver.mli"
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

(* -------------------------------------------------------------------------- *)
(* --- Driver for External Files                                          --- *)
(* -------------------------------------------------------------------------- *)

val load_driver : unit -> LogicBuiltins.driver
(** Memoized loading of drivers according to current
    WP options. Finally sets [LogicBuiltins.driver] and returns it. *)
end
module VCS : sig
# 1 "./VCS.mli"
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

(* -------------------------------------------------------------------------- *)
(** Verification Condition Status *)
(* -------------------------------------------------------------------------- *)

(** {2 Prover} *)

type prover =
  | Why3 of Why3Provers.t (* Prover via WHY *)
  | NativeAltErgo (* Direct Alt-Ergo *)
  | NativeCoq     (* Direct Coq and Coqide *)
  | Qed           (* Qed Solver *)
  | Tactical      (* Interactive Prover *)

type mode =
  | BatchMode (* Only check scripts *)
  | EditMode  (* Edit then check scripts *)
  | FixMode   (* Try check script, then edit script on non-success *)

module Pset : Set.S with type elt = prover
module Pmap : Map.S with type key = prover

val name_of_prover : prover -> string
val title_of_prover : prover -> string
val filename_for_prover : prover -> string
val prover_of_name : string -> prover option
val mode_of_prover_name : string -> mode
val title_of_mode : mode -> string

val pp_prover : Format.formatter -> prover -> unit
val pp_mode : Format.formatter -> mode -> unit

val cmp_prover : prover -> prover -> int

(* -------------------------------------------------------------------------- *)
(** {2 Config}
    [None] means current WP option default.
    [Some 0] means prover default. *)
(* -------------------------------------------------------------------------- *)

type config = {
  valid : bool ;
  timeout : int option ;
  stepout : int option ;
}

val current : unit -> config (** Current parameters *)
val default : config (** all None *)

val get_timeout : config -> int (** 0 means no-timeout *)
val get_stepout : config -> int (** 0 means no-stepout *)

(** {2 Results} *)

type verdict =
  | NoResult
  | Invalid
  | Unknown
  | Timeout
  | Stepout
  | Computing of (unit -> unit) (* kill function *)
  | Checked
  | Valid
  | Failed

type result = {
  verdict : verdict ;
  cached : bool ;
  solver_time : float ;
  prover_time : float ;
  prover_steps : int ;
  prover_errpos : Lexing.position option ;
  prover_errmsg : string ;
}

val no_result : result
val valid : result
val checked : result
val invalid : result
val unknown : result
val stepout : int -> result
val timeout : int -> result
val computing : (unit -> unit) -> result
val failed : ?pos:Lexing.position -> string -> result
val kfailed : ?pos:Lexing.position -> ('a,Format.formatter,unit,result) format4 -> 'a
val cached : result -> result (** only for true verdicts *)

val result : ?cached:bool -> ?solver:float -> ?time:float -> ?steps:int -> verdict -> result

val is_auto : prover -> bool
val is_verdict : result -> bool
val is_valid: result -> bool
val is_computing: result -> bool
val configure : result -> config
val autofit : result -> bool (** Result that fits the default configuration *)

val pp_result : Format.formatter -> result -> unit
val pp_result_perf : Format.formatter -> result -> unit

val compare : result -> result -> int (* best is minimal *)
val merge : result -> result -> result
val choose : result -> result -> result
val best : result list -> result

val dkey_no_time_info: Wp_parameters.category
val dkey_no_step_info: Wp_parameters.category
val dkey_no_goals_info: Wp_parameters.category
val dkey_no_cache_info: Wp_parameters.category
val dkey_success_only: Wp_parameters.category
end
module Tactical : sig
# 1 "./Tactical.mli"
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

(* -------------------------------------------------------------------------- *)
(** Tactical *)
(* -------------------------------------------------------------------------- *)

open Lang.F
open Conditions

(** {2 Tactical Selection} *)

type clause = Goal of pred | Step of step
type process = sequent -> (string * sequent) list
type status =
  | Not_applicable
  | Not_configured
  | Applicable of process

type selection =
  | Empty
  | Clause of clause
  | Inside of clause * term
  | Compose of compose

and compose = private
  | Cint of Integer.t
  | Range of int * int
  | Code of term * string * selection list

val int : int -> selection
val cint : Integer.t -> selection
val range : int -> int -> selection
val compose : string -> selection list -> selection
val get_int : selection -> int option
val destruct : selection -> selection list

val head : clause -> pred
val is_empty : selection -> bool
val selected : selection -> term
val subclause : clause -> pred -> bool
(** When [subclause clause p], we have [clause = Step H] and [H -> p],
    or [clause = Goal G] and [p -> G]. *)

(** Debug only *)
val pp_clause : Format.formatter -> clause -> unit

(** Debug only *)
val pp_selection : Format.formatter -> selection -> unit

(** {2 Tactical Parameters} *)

type 'a field

module Fmap :
sig
  type t
  val create : unit -> t
  val get : t -> 'a field -> 'a (** raises Not_found if absent *)
  val set : t -> 'a field -> 'a -> unit
end

(** {2 Tactical Parameter Editors} *)

type 'a named = { title : string ; descr : string ; vid : string ; value : 'a }
type 'a range = { vmin : 'a option ; vmax : 'a option ; vstep : 'a }
type 'a browser = ('a named -> unit) -> selection -> unit

type parameter =
  | Checkbox of bool field
  | Spinner  of int field * int range
  | Composer of selection field * (Lang.F.term -> bool)
  | Selector : 'a field * 'a named list * ('a -> 'a -> bool) -> parameter
  | Search : 'a named option field * 'a browser * (string -> 'a) -> parameter

val ident : 'a field -> string
val default : 'a field -> 'a
val signature : 'a field -> 'a named

val checkbox :
  id:string -> title:string -> descr:string ->
  ?default:bool ->
  unit -> bool field * parameter
(** Unless specified, default is [false]. *)

val spinner :
  id:string -> title:string -> descr:string ->
  ?default:int ->
  ?vmin:int -> ?vmax:int -> ?vstep:int ->
  unit -> int field * parameter
(** Unless specified, default is [vmin] or [0] or [vmax], whichever fits.
    Range must be non-empty, and default shall fit in. *)

val selector :
  id:string -> title:string -> descr:string ->
  ?default:'a ->
  options:'a named list ->
  ?equal:('a -> 'a -> bool) ->
  unit -> 'a field * parameter
(** Unless specified, default is head option.
    Default equality is [(=)].
    Options must be non-empty. *)

val composer :
  id:string -> title:string -> descr:string ->
  ?default:selection ->
  ?filter:(Lang.F.term -> bool) ->
  unit -> selection field * parameter
(** Unless specified, default is Empty selection. *)

val search :
  id:string -> title:string -> descr:string ->
  browse:('a browser) ->
  find:(string -> 'a) ->
  unit -> 'a named option field * parameter
(** Search field.
    - [browse s n] is the lookup function, used in the GUI only.
       Shall returns at most [n] results applying to selection [s].
    - [find n] is used at script replay, and shall retrieve the
       selected item's [id] later on. *)

type 'a formatter = ('a,Format.formatter,unit) format -> 'a

class type feedback =
  object
    (** Global fresh variable pool *)
    method pool : pool

    (** Interactive mode.
        If [false] the GUI is not activated.
        Hence, detailed feedback is not reported to the user. *)
    method interactive : bool

    method get_title : string
    (** Retrieve the title *)

    method has_error : bool
    (** Retrieve the errors *)

    method set_title : 'a. 'a formatter
    (** Update the title {i wrt} current selection & tuning *)

    method set_descr : 'a. 'a formatter
    (** Add a short description {i wrt} current selection & tuning *)

    method set_error : 'a. 'a formatter
    (** Mark the current configuration as invalid *)

    method update_field :
      'a. ?enabled:bool -> ?title:string -> ?tooltip:string ->
      ?range:bool -> ?vmin:int -> ?vmax:int ->
      ?filter:(Lang.F.term -> bool) -> 'a field -> unit
    (** Update field parameters *)

  end

(** {2 Tactical Utilities} *)

val at : selection -> int option
val mapi : (int -> int -> 'a -> 'b) -> 'a list -> 'b list
val insert : ?at:int -> (string * pred) list -> process
val replace : at:int -> (string * condition) list -> process
val split : (string * pred) list -> process
val rewrite : ?at:int -> (string * pred * term * term) list -> process
(** For each pattern [(descr,guard,src,tgt)] replace [src] with [tgt]
    under condition [guard], inserted in position [at]. *)

(** {2 Tactical Plug-in} *)

class type tactical =
  object
    method id : string
    method title : string
    method descr : string
    method params : parameter list
    method reset : unit
    method get_field : 'a. 'a field -> 'a
    method set_field : 'a. 'a field -> 'a -> unit
    method select : feedback -> selection -> status
  end

class virtual make :
  id:string -> title:string -> descr:string -> params:parameter list ->
  object
    method id : string
    method reset : unit
    method get_field : 'a. 'a field -> 'a
    method set_field : 'a. 'a field -> 'a -> unit

    method title : string
    method descr : string
    method params : parameter list

    method reset : unit
    (** Reset all parameters to default *)

    method virtual select : feedback -> selection -> status
    (** Shall return [Applicable] or [Not_configured]
        if the tactic might apply to the selection.
        Hints can be provided here, if appropriate.

        The continuation [f] returned with [Applicable f] shall generates
        sub-goals {i wrt} to the given selection and current field values.

        @raise Exit,Not_found is like returning Not_applicable. *)
  end

(** {2 Composer Factory} *)

class type composer =
  object
    method id : string
    method group : string
    method title : string
    method descr : string
    method arity : int
    method filter : term list -> bool
    method compute : term list -> term
  end

(** {2 Global Registry} *)

type t = tactical

val register : #tactical -> unit
val export : #tactical -> tactical (** Register and returns the tactical *)
val lookup : id:string -> tactical
val iter : (tactical -> unit) -> unit

val add_composer : #composer -> unit
val iter_composer : (composer -> unit) -> unit
end
module Strategy : sig
# 1 "./Strategy.mli"
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

(** {2 Term & Predicate Selection} *)

open Lang.F
open Conditions
open Tactical

val occurs_x : var -> term -> bool
val occurs_y : var -> pred -> bool
val occurs_e : term -> term -> bool
val occurs_p : term -> pred -> bool
val occurs_q : pred -> pred -> bool

(** Lookup the first occurrence of term in the sequent and returns
    the associated selection. Returns [Empty] is not found.
    Goal is lookup first. *)
val select_e : sequent -> term -> selection

(** Same as [select_e] but for a predicate. *)
val select_p : sequent -> pred -> selection

(** {2 Strategy} *)

type argument = ARG: 'a field * 'a -> argument

type strategy = {
  priority : float ;
  tactical : tactical ;
  selection : selection ;
  arguments : argument list ;
}

class pool :
  object
    method add : strategy -> unit
    method sort : strategy array
  end

class type heuristic =
  object
    method id : string
    method title : string
    method descr : string
    method search : (strategy -> unit) -> sequent -> unit
  end

val register : #heuristic -> unit
val export : #heuristic -> heuristic
val lookup : id:string -> heuristic
val iter : (heuristic -> unit) -> unit

(** {2 Factory} *)

type t = strategy
val arg : 'a field -> 'a -> argument
val make : tactical ->
  ?priority:float -> ?arguments:argument list -> selection -> strategy

(**/**)

(* To be used only when applying the tactical *)

val set_arg : tactical -> argument -> unit
val set_args : tactical -> argument list -> unit
end
module Auto : sig
# 1 "./Auto.mli"
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

open Tactical
open Strategy

(* -------------------------------------------------------------------------- *)
(** {2 Basic Strategies}
    It is always safe to apply strategies to any goal. *)
(* -------------------------------------------------------------------------- *)

val array : ?priority:float -> selection -> strategy
val choice : ?priority:float -> selection -> strategy
val absurd : ?priority:float -> selection -> strategy
val contrapose : ?priority:float -> selection -> strategy
val compound : ?priority:float -> selection -> strategy
val cut : ?priority:float -> ?modus:bool -> selection -> strategy
val filter : ?priority:float -> ?anti:bool -> unit -> strategy
val havoc : ?priority:float -> havoc:selection -> strategy
val separated : ?priority:float -> selection -> strategy
val instance : ?priority:float -> selection -> selection list -> strategy
val lemma : ?priority:float -> ?at:selection -> string -> selection list -> strategy
val intuition : ?priority:float -> selection -> strategy
val range : ?priority:float -> selection -> vmin:int -> vmax:int -> strategy
val split : ?priority:float -> selection -> strategy
val definition : ?priority:float -> selection -> strategy

(* -------------------------------------------------------------------------- *)
(** {2 Registered Heuristics} *)
(* -------------------------------------------------------------------------- *)

val auto_split : Strategy.heuristic
val auto_range : Strategy.heuristic

module Range :
sig
  type rg
  val compute : Conditions.sequence -> rg
  val ranges : rg -> (int * int) Lang.F.Tmap.t
  val bounds : rg -> (int option * int option) Lang.F.Tmap.t
end

(* -------------------------------------------------------------------------- *)
(** {2 Trusted Tactical Process}
    Tacticals with hand-written process are not safe.
    However, the combinators below are guarantied to be sound. *)
(* -------------------------------------------------------------------------- *)

(** Find a contradiction. *)
val t_absurd : process

(** Keep goal unchanged. *)
val t_id : process

(** Apply a description to a leaf goal. Same as [t_descr "..." t_id]. *)
val t_finally : string -> process

(** Apply a description to each sub-goal *)
val t_descr : string -> process -> process

(** Split with [p] and [not p]. *)
val t_split : ?pos:string -> ?neg:string -> Lang.F.pred -> process

(** Prove condition [p] and use-it as a forward hypothesis. *)
val t_cut : ?by:string -> Lang.F.pred -> process -> process

(** Case analysis: [t_case p a b] applies process [a] under hypothesis [p]
    and process [b] under hypothesis [not p]. *)
val t_case : Lang.F.pred -> process -> process -> process

(** Complete analysis: applies each process under its guard, and proves that
    all guards are complete. *)
val t_cases : ?complete:string -> (Lang.F.pred * process) list -> process

(** Apply second process to every goal generated by the first one. *)
val t_chain : process -> process -> process

(** @raise Invalid_argument when range is empty *)
val t_range : Lang.F.term -> int -> int ->
  upper:process -> lower:process -> range:process -> process

(** Prove [src=tgt] then replace [src] by [tgt]. *)
val t_replace :
  ?equal:string -> src:Lang.F.term -> tgt:Lang.F.term -> process -> process

(**************************************************************************)
end
module VC : sig
# 1 "./VC.mli"
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

(* -------------------------------------------------------------------------- *)
(* --- Verification Conditions Interface                                  --- *)
(* -------------------------------------------------------------------------- *)

open VCS

(** {2 Proof Obligations} *)

type t (** elementary proof obligation *)

val get_id : t -> string
val get_model : t -> WpContext.model
val get_scope : t -> WpContext.scope
val get_context : t -> WpContext.context
val get_description : t -> string
val get_property : t -> Property.t
val get_result : t -> prover -> result
val get_results : t -> (prover * result) list
val get_logout : t -> prover -> string (** only file name, might not exists *)
val get_logerr : t -> prover -> string (** only file name, might not exists *)
val get_sequent : t -> Conditions.sequent
val get_formula: t -> Lang.F.pred
val is_trivial : t -> bool
val is_proved : t -> bool

(** {2 Database}
    Notice that a property or a function have no proof obligation until you
    explicitly generate them {i via} the [generate_xxx] functions below.
*)

val clear : unit -> unit
val proof : Property.t -> t list
(** List of proof obligations computed for a given property. Might be empty if you
    don't have used one of the generators below. *)

val remove : Property.t -> unit
val iter_ip : (t -> unit) -> Property.t -> unit
val iter_kf : (t -> unit) -> ?bhv:string list -> Kernel_function.t -> unit

(** {2 Generators}
    The generated VCs are also added to the database, so they can be
    accessed later. The default value for [model] is what has been
    given on the command line ([-wp-model] option)
*)

val generate_ip : ?model:string -> Property.t -> t Bag.t
val generate_kf : ?model:string -> ?bhv:string list -> Kernel_function.t -> t Bag.t
val generate_call : ?model:string -> Cil_types.stmt -> t Bag.t

(** {2 Prover Interface} *)

val prove : t ->
  ?config:config ->
  ?mode:mode ->
  ?start:(t -> unit) ->
  ?progress:(t -> string -> unit) ->
  ?result:(t -> prover -> result -> unit) ->
  prover -> bool Task.task
(** Returns a ready-to-schedule task. *)

val spawn : t ->
  ?config:config ->
  ?start:(t -> unit) ->
  ?progress:(t -> string -> unit) ->
  ?result:(t -> prover -> result -> unit) ->
  ?success:(t -> prover option -> unit) ->
  ?pool:Task.pool ->
  (mode * prover) list -> unit
(** Same as [prove] but schedule the tasks into the global server returned
    by [server] function below.

    The first succeeding prover cancels the other ones. *)

val server : ?procs:int -> unit -> Task.server
(** Default number of parallel tasks is given by [-wp-par] command-line option.
    The returned server is global to Frama-C, but the number of parallel task
    allowed will be updated to fit the [~procs] or command-line options. *)

val command : ?provers:Why3.Whyconf.prover list -> ?tip:bool -> t Bag.t -> unit
(** Run the provers with the command-line interface.
    If [~provers] is set, it is used for computing the list of provers to spawn.
    If [~tip] is set, it is used to compute the script execution mode. *)

(* -------------------------------------------------------------------------- *)
end
module Wpo : sig
# 1 "./wpo.mli"
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

open LogicUsage
open VCS
open Cil_types
open Cil_datatype
open WpPropId

type index =
  | Axiomatic of string option
  | Function of kernel_function * string option

(* ------------------------------------------------------------------------ *)
(**{1 Proof Obligations}                                                    *)
(* ------------------------------------------------------------------------ *)

module DISK :
sig
  val cache_log : pid:prop_id -> model:WpContext.model ->
    prover:prover -> result:result -> string
  val pretty : pid:prop_id -> model:WpContext.model ->
    prover:prover -> result:result -> Format.formatter -> unit
  val file_kf : kf:kernel_function -> model:WpContext.model -> prover:prover -> string
  val file_goal : pid:prop_id -> model:WpContext.model -> prover:prover -> string
  val file_logout : pid:prop_id -> model:WpContext.model -> prover:prover -> string
  val file_logerr : pid:prop_id -> model:WpContext.model -> prover:prover -> string
end

module GOAL :
sig
  open Lang
  type t = {
    mutable time : float ;
    mutable simplified : bool ;
    mutable sequent : Conditions.sequent ;
    mutable obligation : F.pred ;
  }
  val dummy : t
  val trivial : t
  val is_trivial : t -> bool
  val make : Conditions.sequent -> t
  val compute_proof : t -> F.pred
  val compute_descr : t -> Conditions.sequent
  val get_descr : t -> Conditions.sequent
  val compute : t -> unit
  val qed_time : t -> float
end

module VC_Lemma :
sig

  type t = {
    lemma : Definitions.dlemma ;
    depends : logic_lemma list ;
    mutable sequent : Conditions.sequent option ;
  }

  val is_trivial : t -> bool
  val cache_descr : t -> (prover * result) list -> string

end

module VC_Annot :
sig

  type t = {
    axioms : Definitions.axioms option ;
    goal : GOAL.t ;
    tags : Splitter.tag list ;
    warn : Warning.t list ;
    deps : Property.Set.t ;
    path : Stmt.Set.t ;
    effect : (stmt * effect_source) option ;
  }

  val resolve : t -> bool
  val is_trivial : t -> bool
  val cache_descr : pid:prop_id -> t -> (prover * result) list -> string

end

(* ------------------------------------------------------------------------ *)
(**{1 Proof Obligations}                                                    *)
(* ------------------------------------------------------------------------ *)

type formula =
  | GoalLemma of VC_Lemma.t
  | GoalAnnot of VC_Annot.t

type po = t and t = {
    po_gid   : string ;  (** goal identifier *)
    po_leg   : string ; (** legacy goal identifier *)
    po_sid   : string ;  (** goal short identifier (without model) *)
    po_name  : string ;  (** goal informal name *)
    po_idx   : index ;   (** goal index *)
    po_model : WpContext.model ;
    po_pid   : WpPropId.prop_id ; (* goal target property *)
    po_formula : formula ; (* proof obligation *)
  }

module S : Datatype.S_with_collections with type t = po
module Index : Map.OrderedType with type t = index
module Gmap : FCMap.S with type key = index

(** Dynamically exported
    @since Nitrogen-20111001
*)
val get_gid: t -> string

(** Dynamically exported
    @since Oxygen-20120901
*)
val get_property: t -> Property.t
val get_index : t -> index
val get_label : t -> string
val get_model : t -> WpContext.model
val get_scope : t -> WpContext.scope
val get_context : t -> WpContext.context
val get_file_logout : t -> prover -> string (** only filename, might not exists *)
val get_file_logerr : t -> prover -> string (** only filename, might not exists *)

val get_files : t -> (string * string) list

val qed_time : t -> float

val clear : unit -> unit
val remove : t -> unit
val on_remove : (t -> unit) -> unit

val add : t -> unit
val age : t -> int (* generation *)

val reduce : t -> bool (** tries simplification *)
val resolve : t -> bool (** tries simplification and set result if valid *)
val set_result : t -> prover -> result -> unit
val clear_results : t -> unit

val compute : t -> Definitions.axioms option * Conditions.sequent

val has_verdict : t -> prover -> bool
val get_result : t -> prover -> result
val get_results : t -> (prover * result) list
val get_proof : t -> bool * Property.t
val is_trivial : t -> bool (** do not tries simplification, do not check prover results *)
val is_proved : t -> bool (** do not tries simplification, check prover results *)
val is_unknown : t -> bool
val warnings : t -> Warning.t list

(** [true] if the result is valid. Dynamically exported.
    @since Nitrogen-20111001
*)
val is_valid: result -> bool

val get_time: result -> float
val get_steps: result -> int

val is_tactic : t -> bool

val iter :
  ?ip:Property.t ->
  ?index:index ->
  ?on_axiomatics:(string option -> unit) ->
  ?on_behavior:(kernel_function -> string option -> unit) ->
  ?on_goal:(t -> unit) ->
  unit -> unit

(** Dynamically exported.
    @since Nitrogen-20111001
*)
val iter_on_goals: (t -> unit) -> unit

(** All POs related to a given property.
    Dynamically exported
    @since Oxygen-20120901
*)
val goals_of_property: Property.t -> t list

val bar : string
val kf_context : index -> Description.kf
val pp_index : Format.formatter -> index -> unit
val pp_warnings : Format.formatter -> Warning.t list -> unit
val pp_depend : Format.formatter -> Property.t -> unit
val pp_dependency : Description.kf -> Format.formatter -> Property.t -> unit
val pp_dependencies : Description.kf -> Format.formatter -> Property.t list -> unit
val pp_goal : Format.formatter -> t -> unit
val pp_title : Format.formatter -> t -> unit
val pp_logfile : Format.formatter -> t -> prover -> unit

val pp_axiomatics : Format.formatter -> string option -> unit
val pp_function : Format.formatter -> Kernel_function.t -> string option -> unit
val pp_goal_flow : Format.formatter -> t -> unit

(** Dynamically exported. *)
val prover_of_name : string -> prover option
end
module ProverTask : sig
# 1 "./ProverTask.mli"
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

(* -------------------------------------------------------------------------- *)
(* --- Library for Running Provers                                        --- *)
(* -------------------------------------------------------------------------- *)

class printer : Format.formatter -> string ->
  object
    method paragraph : unit
    method lines : unit
    method section : string -> unit
    method hline : unit
    method printf : 'a. ('a,Format.formatter,unit) format -> 'a
  end

val pp_file : message:string -> file:string -> unit

(** never fails *)
class type pattern =
  object
    method get_after : ?offset:int -> int -> string
    (** [get_after ~offset:p k] returns the end of the message
                starting [p] characters after the end of group [k]. *)
    method get_string : int -> string
    method get_int : int -> int
    method get_float : int -> float
  end

val p_group : string -> string (** Put pattern in group [\(p\)] *)
val p_int : string (** Int group pattern [\([0-9]+\)] *)
val p_float : string (** Float group pattern [\([0-9.]+\)] *)
val p_string : string (** String group pattern ["\(...\)"] *)
val p_until_space : string (** No space group pattern "\\([^ \t\n]*\\)" *)

val location : string -> int -> Lexing.position

val timeout : int option -> int
val stepout : int option -> int
type logs = [ `OUT | `ERR | `BOTH ]

class virtual command : string ->
  object
    method command : string list
    method pretty : Format.formatter -> unit
    method set_command : string -> unit
    method add : string list -> unit
    method add_int : name:string -> value:int -> unit
    method add_positive : name:string -> value:int -> unit
    method add_float : name:string -> value:float -> unit
    method add_parameter : name:string -> (unit -> bool) -> unit
    method add_list : name:string -> string list -> unit
    method timeout : int -> unit
    method validate_time : (float -> unit) -> unit
    method validate_pattern : ?logs:logs -> ?repeat:bool ->
      Str.regexp -> (pattern -> unit) -> unit
    method run : ?echo:bool -> ?logout:string -> ?logerr:string ->
      unit -> int Task.task

  end

val server : ?procs:int -> unit -> Task.server

val schedule : 'a Task.task -> unit

val spawn :
  ?monitor:('a option -> unit) ->
  ?pool:Task.pool ->
  ('a * bool Task.task) list -> unit
(** Spawn all the tasks over the server and retain the first 'validated' one.
    The callback [monitor] is called with [Some] at first success, and [None]
    if none succeed.
    An option [pool] task can be passed to register the associated threads. *)
end
module Prover : sig
# 1 "./prover.mli"
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

open VCS

(* -------------------------------------------------------------------------- *)
(* --- Prover Implementation against Task API                             --- *)
(* -------------------------------------------------------------------------- *)

val prove : Wpo.t ->
  ?config:config ->
  ?mode:mode ->
  ?start:(Wpo.t -> unit) ->
  ?progress:(Wpo.t -> string -> unit) ->
  ?result:(Wpo.t -> prover -> result -> unit) ->
  prover -> bool Task.task

val spawn : Wpo.t ->
  delayed:bool ->
  ?config:config ->
  ?start:(Wpo.t -> unit) ->
  ?progress:(Wpo.t -> string -> unit) ->
  ?result:(Wpo.t -> prover -> result -> unit) ->
  ?success:(Wpo.t -> prover option -> unit) ->
  ?pool:Task.pool ->
  (mode * prover) list -> unit
end
