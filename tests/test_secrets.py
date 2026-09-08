"""What a runtime does with the values it was given.

They arrive apart from every service's state and leave only through ``resolve``,
for one use, named destination first. The behaviour pinned here is the same
behaviour hkp-node, hkp-rt and the browser pin: a board written against one
runtime has to open against another, which makes these the shared contract
rather than this runtime's own arrangement.
"""

from __future__ import annotations

from hkp.secrets import (
    SecretVault,
    audience_permits,
    destination_host,
    read_secrets_payload,
    referenced_secrets,
    resolve_credential,
)


def vault_of(payload: dict) -> SecretVault:
    vault = SecretVault()
    vault.replace(read_secrets_payload(payload))
    return vault


def test_substitutes_the_value_the_reference_names() -> None:
    resolved = vault_of({"mail": {"value": "hunter2"}}).resolve(
        {"pass": "{{secret.mail}}"}, "imap.example.com:993"
    )

    assert resolved.value == {"pass": "hunter2"}
    assert resolved.missing == []
    assert resolved.refused == []


def test_substitutes_inside_a_larger_string_and_anywhere_nested() -> None:
    resolved = vault_of({"api": {"value": "sk-1"}}).resolve(
        {"headers": {"Authorization": "Bearer {{secret.api}}"}},
        "https://api.example.com/v1",
    )

    assert resolved.value["headers"]["Authorization"] == "Bearer sk-1"


def test_tolerates_whitespace_and_treats_dots_as_part_of_the_alias() -> None:
    resolved = vault_of({"gmail.imap": {"value": "v"}}).resolve(
        "{{ secret.gmail.imap }}", "imap.gmail.com"
    )

    assert resolved.value == "v"


def test_an_alias_it_was_not_given_resolves_to_nothing_and_is_named() -> None:
    resolved = vault_of({}).resolve({"pass": "{{secret.absent}}"}, "example.com")

    assert resolved.value == {"pass": ""}
    assert resolved.missing == ["absent"]


def test_leaves_a_value_with_no_reference_alone() -> None:
    assert vault_of({}).resolve("literal", "example.com").value == "literal"


def test_releases_a_secret_to_a_host_it_is_bound_to() -> None:
    resolved = vault_of(
        {"slack": {"value": "xoxb", "audience": ["hooks.slack.com"]}}
    ).resolve("{{secret.slack}}", "https://hooks.slack.com/services/x")

    assert resolved.value == "xoxb"
    assert resolved.refused == []


def test_withholds_it_from_anywhere_else_and_says_so() -> None:
    resolved = vault_of(
        {"slack": {"value": "xoxb", "audience": ["hooks.slack.com"]}}
    ).resolve("{{secret.slack}}", "https://evil.example/?p=1")

    assert resolved.value == ""
    assert [(r.alias, r.to) for r in resolved.refused] == [("slack", "evil.example")]


def test_an_entry_with_no_audience_is_unconstrained() -> None:
    resolved = vault_of({"any": {"value": "v"}}).resolve(
        "{{secret.any}}", "https://anywhere.example"
    )

    assert resolved.value == "v"


def test_a_subdomain_wildcard_does_not_cover_the_bare_domain() -> None:
    vault = vault_of({"k": {"value": "v", "audience": ["*.example.com"]}})

    assert vault.resolve("{{secret.k}}", "api.example.com").value == "v"
    assert len(vault.resolve("{{secret.k}}", "example.com").refused) == 1
    assert audience_permits(["*.example.com"], "example.com") is False


def test_reads_a_host_out_of_whatever_shape_a_caller_holds() -> None:
    assert destination_host("https://api.example.com/v1?q=1") == "api.example.com"
    assert destination_host("imap.example.com:993") == "imap.example.com"
    assert destination_host("API.Example.COM") == "api.example.com"
    assert destination_host("  example.com  ") == "example.com"
    assert destination_host("") == ""
    assert destination_host(None) == ""


def test_releases_nothing_without_a_destination_to_check_against() -> None:
    resolved = vault_of({"mail": {"value": "hunter2"}}).resolve("{{secret.mail}}", "")

    assert resolved.missing == ["mail"]


def test_names_what_it_holds_and_says_nothing_else_about_it() -> None:
    vault = vault_of({"a": {"value": "1"}, "b": {"value": "2"}})

    assert sorted(vault.aliases()) == ["a", "b"]


def test_a_partial_push_leaves_the_rest_in_place() -> None:
    vault = vault_of({"a": {"value": "1"}, "b": {"value": "2"}})
    vault.merge(read_secrets_payload({"b": {"value": "changed"}}))

    assert vault.resolve("{{secret.a}} {{secret.b}}", "x.example").value == "1 changed"


def test_a_payload_may_name_a_value_directly_or_describe_it() -> None:
    short = read_secrets_payload({"a": "v"})
    assert short["a"].value == "v"
    assert short["a"].audience == []

    long = read_secrets_payload({"a": {"value": "v", "audience": ["h.example"]}})
    assert long["a"].audience == ["h.example"]


def test_drops_what_it_cannot_read_rather_than_failing_the_request() -> None:
    entries = read_secrets_payload(
        {"a": {"audience": ["x"]}, "b": 7, "c": None, "d": "ok"}
    )

    assert list(entries) == ["d"]
    assert read_secrets_payload("nonsense") == {}


def test_names_every_alias_a_board_refers_to_however_nested() -> None:
    assert sorted(
        referenced_secrets(
            {"a": "{{secret.one}}", "b": [{"c": "x {{secret.two}} y"}], "d": "{{secret.one}}"}
        )
    ) == ["one", "two"]


class TestACredentialInSomethingLarger:
    """A credential is not always a field of its own.

    ``http-client`` carries one as an entry in a free-form header map, inside a
    larger string — and reports that map verbatim, which is what used to write a
    resolved token into the next saved board.
    """

    def test_a_literal_passes_through_with_no_vault_needed(self) -> None:
        out = resolve_credential(
            None, {"Authorization": "Bearer literal"}, "https://api.example.com"
        )

        assert out.problem == ""
        assert out.value == {"Authorization": "Bearer literal"}

    def test_a_reference_with_no_vault_is_never_sent_as_its_own_text(self) -> None:
        out = resolve_credential(
            None, "Bearer {{secret.api}}", "https://api.example.com"
        )

        assert out.value is None
        assert out.problem == "no secrets available to resolve api"

    def test_a_whole_map_resolves_at_once(self) -> None:
        out = resolve_credential(
            vault_of({"api": {"value": "sk-1"}}),
            {"Authorization": "Bearer {{secret.api}}", "Accept": "application/json"},
            "https://api.example.com/v1",
        )

        assert out.problem == ""
        assert out.value == {
            "Authorization": "Bearer sk-1",
            "Accept": "application/json",
        }

    def test_resolves_nothing_at_all_when_one_entry_may_not_go_there(self) -> None:
        # Not a half-filled map: a caller handed one might send it anyway.
        out = resolve_credential(
            vault_of({"api": {"value": "sk-1", "audience": ["api.example.com"]}}),
            {"Authorization": "Bearer {{secret.api}}", "Accept": "application/json"},
            "https://evil.example/",
        )

        assert out.value is None
        assert out.problem == "api may not be sent to evil.example"
