//! Headless acceptance test: a client and the server exchange real
//! protocol messages over 100 ticks of the simulation.

use jbm_core::physics::MoveInput;
use jbm_core::protocol::{ClientMsg, PROTOCOL_VERSION, ServerMsg};
use jbm_testkit::HeadlessServer;

/// One idle movement input per tick; the physics steps but nothing moves.
fn input(seq: u64) -> ClientMsg {
    ClientMsg::PlayerInput {
        seq,
        input: MoveInput::IDLE,
        pitch: 0.0,
    }
}

#[test]
fn hundred_tick_exchange_of_inputs_pings_and_ticks() {
    let mut server = HeadlessServer::new();
    let mut client = server.connect("alice");

    server.step(1);
    assert!(matches!(
        client.drain().first(),
        Some(ServerMsg::Welcome { .. })
    ));

    let mut expected_tick: Option<u64> = None;
    for i in 0..100u64 {
        client.send(input(i + 1));
        client.send(ClientMsg::Ping { nonce: i });
        server.step(1);
        let messages = client.drain();
        match messages.as_slice() {
            [
                ServerMsg::Pong { nonce, .. },
                ServerMsg::PlayerState { .. },
                ServerMsg::Tick { tick },
            ] if *nonce == i => match expected_tick {
                None => expected_tick = Some(*tick),
                Some(prev) => {
                    assert_eq!(*tick, prev + 1);
                    expected_tick = Some(*tick);
                }
            },
            other => panic!("tick {i}: expected [Pong, PlayerState, Tick], got {other:?}"),
        }
    }

    let players: Vec<_> = server.server().players().collect();
    assert_eq!(players.len(), 1);
    assert_eq!(players[0].input_count, 100);
    assert_eq!(players[0].last_seq, 100);
    // Idle inputs on a protocol-only server (no cubes): the unloaded rule
    // keeps the player exactly at spawn.
    assert_eq!(players[0].state.position.x, 0.5);
    assert_eq!(players[0].state.position.z, 0.5);
}

#[test]
fn two_clients_get_distinct_ids_and_every_tick() {
    let mut server = HeadlessServer::new();
    let mut a = server.connect("alice");
    let mut b = server.connect("bob");

    server.step(1);
    let _ = a.drain();
    let _ = b.drain();
    let a_id = a.player_id().expect("alice welcomed");
    let b_id = b.player_id().expect("bob welcomed");
    assert_ne!(a_id, b_id);

    for _ in 0..5 {
        server.step(1);
        for (name, client) in [("alice", &mut a), ("bob", &mut b)] {
            let messages = client.drain();
            assert_eq!(
                messages.len(),
                1,
                "{name} should receive exactly the tick message, got {messages:?}"
            );
            assert!(matches!(messages[0], ServerMsg::Tick { .. }));
        }
    }
}

#[test]
fn wrong_protocol_version_is_rejected_and_dropped() {
    let mut server = HeadlessServer::new();
    let mut client = server.connect_raw();
    client.send(ClientMsg::Hello {
        protocol_version: PROTOCOL_VERSION + 1,
        name: "mallory".to_owned(),
    });

    server.step(1);
    assert!(matches!(
        client.drain().as_slice(),
        [ServerMsg::Rejected { .. }]
    ));

    // The connection is gone: no players, no further messages, no panic.
    server.step(1);
    assert_eq!(server.server().players().count(), 0);
    assert!(client.drain().is_empty());
}

#[test]
fn dropped_link_removes_the_player_without_panicking() {
    let mut server = HeadlessServer::new();
    let mut client = server.connect("alice");
    server.step(1);
    let _ = client.drain();
    assert_eq!(server.server().players().count(), 1);

    drop(client);
    server.step(1);
    assert_eq!(server.server().players().count(), 0);
}

#[test]
fn flood_of_pings_is_answered_across_four_ticks_in_order() {
    let mut server = HeadlessServer::new();
    let mut client = server.connect("flood");
    server.step(1);
    let _ = client.drain();

    const FLOOD: u64 = 1000;
    for nonce in 0..FLOOD {
        client.send(ClientMsg::Ping { nonce });
    }

    // 256 messages per connection per tick: 1000 pings span 4 ticks,
    // every one answered, in order.
    let mut next_nonce = 0;
    for expected in [256, 256, 256, 232] {
        server.step(1);
        let messages = client.drain();
        let pongs: Vec<&ServerMsg> = messages
            .iter()
            .filter(|m| matches!(m, ServerMsg::Pong { .. }))
            .collect();
        assert_eq!(pongs.len(), expected, "pongs answered in one tick");
        assert!(matches!(messages.last(), Some(ServerMsg::Tick { .. })));
        for msg in pongs {
            let ServerMsg::Pong { nonce, .. } = msg else {
                unreachable!("filtered to pongs above")
            };
            assert_eq!(*nonce, next_nonce);
            next_nonce += 1;
        }
    }
    assert_eq!(next_nonce, FLOOD);
}
