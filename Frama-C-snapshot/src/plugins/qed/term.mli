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

(** Logic expressions *)



module IntPairs: sig  

   type t 

   
   end

module Make
    ( ADT : Logic.Data )
    ( Field : Logic.Field )
    ( Fun : Logic.Function )
  :
  sig


    (** Logic API *)
    include Logic.Term with module ADT = ADT
                        and module Field = Field
                        and module Fun = Fun

   type dag_node = {

    mutable var_node : var;
    mutable dependencies : var list;
    mutable stmt : Cil_types.stmt option;
    mutable flag : bool

   }



    (*module VarMap = Map.Make(struct type t = var let compare = Pervasives.compare end)*)

    val dag:  dag_node  list ref 
  
    val dag_before_instrument : dag_node list ref

    val transform_term : term -> term

    val add_node_: term -> Cil_types.stmt -> unit

    val add_node_eq : var -> var -> unit

    val add_node: term -> String.t -> unit

    val update_namespace_term: term -> unit

    val add_name_fun: String.t -> unit

    val dag_copy: unit -> unit

    val dag_copy_replace: unit -> unit

    (*val update_term_name : term list -> int NameSpace.t -> (term list) * int NameSpace.t*)

    val update_term_name : term list -> term list



    (** Prints term in debug mode. *)
    val debug : Format.formatter -> term -> unit

    (** {2 Global State}
        One given [term] has valid meaning only for one particular state. *)

    type state (** Hash-consing, cache, rewriting rules, etc. *)
    val create : unit -> state
    (** Create a new fresh state. Local state is not modified. *)

    val get_state : unit -> state (** Return local state. *)
    val set_state : state -> unit (** Update local state. *)
    val clr_state : state -> unit (** Clear local state. *)

    val in_state : state -> ('a -> 'b) -> 'a -> 'b
    (** execute in a particular state. *)

    val rebuild_in_state : state -> ?cache:term Tmap.t -> term -> term * term Tmap.t
    (** rebuild a term in the given state *)

    (** Register a constant in the global state. *)
    val constant : term -> term

    (** {2 Context Release} *)

    val release : unit -> unit
    (** Clear caches and checks. Global builtins are kept. *)

  end
