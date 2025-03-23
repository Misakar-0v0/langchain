"""Test Anthropic API wrapper."""

from typing import List

from langchain_core.callbacks import (
    CallbackManager,
)
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, FunctionMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from langchain_community.chat_models.litellm import ChatLiteLLM
from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler


def test_litellm_call() -> None:
    """Test valid call to litellm."""
    chat = ChatLiteLLM(  # type: ignore[call-arg]
        model="test",
    )
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def test_litellm_generate() -> None:
    """Test generate method of anthropic."""
    chat = ChatLiteLLM(model="test")  # type: ignore[call-arg]
    chat_messages: List[List[BaseMessage]] = [
        [HumanMessage(content="How many toes do dogs have?")]
    ]
    messages_copy = [messages.copy() for messages in chat_messages]
    result: LLMResult = chat.generate(chat_messages)
    assert isinstance(result, LLMResult)
    for response in result.generations[0]:
        assert isinstance(response, ChatGeneration)
        assert isinstance(response.text, str)
        assert response.text == response.message.content
    assert chat_messages == messages_copy


def test_litellm_streaming() -> None:
    """Test streaming tokens from anthropic."""
    chat = ChatLiteLLM(model="test", streaming=True)  # type: ignore[call-arg]
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def test_litellm_streaming_callback() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    chat = ChatLiteLLM(  # type: ignore[call-arg]
        model="test",
        streaming=True,
        callback_manager=callback_manager,
        verbose=True,
    )
    message = HumanMessage(content="Write me a sentence with 10 words.")
    chat.invoke([message])
    assert callback_handler.llm_streams > 1


def test_litellm_tool_call() -> None:
    """Test tool calling functionality."""
    chat = ChatLiteLLM(model="test")  # type: ignore[call-arg]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather in a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    chat_with_tools = chat.bind_tools(tools)
    message = HumanMessage(content="What's the weather in San Francisco?")
    response = chat_with_tools.invoke([message])
    assert isinstance(response, AIMessage)
    assert "tool_calls" in response.additional_kwargs


def test_litellm_message_conversion() -> None:
    """Test message conversion functionality."""
    chat = ChatLiteLLM(model="test")  # type: ignore[call-arg]
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there!"),
        FunctionMessage(name="get_time", content="The current time is 12:00 PM"),
    ]
    response = chat.invoke(messages)
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def test_litellm_model_params() -> None:
    """Test model configuration parameters."""
    chat = ChatLiteLLM(
        model="test",
        temperature=0.7,
        top_p=0.9,
        max_tokens=100,
        n=2,
    )  # type: ignore[call-arg]
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def test_litellm_retry_mechanism() -> None:
    """Test retry mechanism for API calls."""
    chat = ChatLiteLLM(model="test", max_retries=3)  # type: ignore[call-arg]
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def test_convert_delta_to_message_chunk() -> None:
    """Test _convert_delta_to_message_chunk method with various scenarios."""
    from langchain_core.messages import (
        AIMessageChunk,
        HumanMessageChunk,
        SystemMessageChunk,
        FunctionMessageChunk,
        ChatMessageChunk,
    )
    from langchain_community.chat_models.litellm import _convert_delta_to_message_chunk

    # Test function_call conversion
    function_call_dict = {
        "role": "assistant",
        "content": None,
        "function_call": {
            "name": "get_weather",
            "arguments": "{\"location\": \"San Francisco\"}"
        }
    }
    chunk = _convert_delta_to_message_chunk(function_call_dict, AIMessageChunk)
    assert isinstance(chunk, AIMessageChunk)
    assert chunk.additional_kwargs["function_call"]["name"] == "get_weather"

    # Test reasoning_content conversion
    reasoning_dict = {
        "role": "assistant",
        "content": "Final answer",
        "reasoning_content": "Step-by-step reasoning"
    }
    chunk = _convert_delta_to_message_chunk(reasoning_dict, AIMessageChunk)
    assert isinstance(chunk, AIMessageChunk)
    assert chunk.additional_kwargs["reasoning_content"] == "Step-by-step reasoning"

    # Test tool_calls conversion
    tool_calls_dict = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "call_1",
            "index": 0,
            "function": {
                "name": "get_weather",
                "arguments": "{\"location\": \"London\"}"
            }
        }]
    }
    chunk = _convert_delta_to_message_chunk(tool_calls_dict, AIMessageChunk)
    assert isinstance(chunk, AIMessageChunk)
    assert len(chunk.tool_call_chunks) == 1
    assert chunk.tool_call_chunks[0].name == "get_weather"
    assert chunk.tool_call_chunks[0].id == "call_1"

    # Test different role conversions
    roles = [
        ("user", HumanMessageChunk),
        ("assistant", AIMessageChunk),
        ("system", SystemMessageChunk),
        ("function", FunctionMessageChunk),
        ("custom", ChatMessageChunk)
    ]

    for role, expected_class in roles:
        content = "Test content"
        if role == "function":
            chunk = _convert_delta_to_message_chunk(
                {"role": role, "content": content, "name": "test_func"},
                expected_class
            )
        else:
            chunk = _convert_delta_to_message_chunk(
                {"role": role, "content": content},
                expected_class
            )
        assert isinstance(chunk, expected_class)
        assert chunk.content == content

    # Test empty content
    empty_dict = {"role": "assistant"}
    chunk = _convert_delta_to_message_chunk(empty_dict, AIMessageChunk)
    assert isinstance(chunk, AIMessageChunk)
    assert chunk.content == ""

    # Test invalid tool_calls
    invalid_tool_calls_dict = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "call_1",
            "function": {}
        }]
    }
    chunk = _convert_delta_to_message_chunk(invalid_tool_calls_dict, AIMessageChunk)
    assert isinstance(chunk, AIMessageChunk)
    assert "tool_calls" in chunk.additional_kwargs


def test_convert_message_to_dict() -> None:
    """Test _convert_message_to_dict method with various scenarios."""
    from langchain_core.messages import (
        AIMessage,
        HumanMessage,
        SystemMessage,
        FunctionMessage,
        ChatMessage,
        ToolMessage
    )
    from langchain_community.chat_models.litellm import _convert_message_to_dict

    # Test human message conversion
    human_msg = HumanMessage(content="Hello")
    result = _convert_message_to_dict(human_msg)
    assert result["role"] == "user"
    assert result["content"] == "Hello"

    # Test AI message with function call
    ai_msg = AIMessage(
        content="Let me check the weather",
        additional_kwargs={
            "function_call": {"name": "get_weather", "arguments": "{\"location\": \"London\"}"}
        }
    )
    result = _convert_message_to_dict(ai_msg)
    assert result["role"] == "assistant"
    assert result["content"] == "Let me check the weather"
    assert result["function_call"]["name"] == "get_weather"

    # Test AI message with tool calls
    ai_tool_msg = AIMessage(
        content="Let me help you",
        additional_kwargs={
            "tool_calls": [{
                "id": "call_1",
                "function": {"name": "get_weather", "arguments": "{\"location\": \"London\"}"}
            }]
        }
    )
    result = _convert_message_to_dict(ai_tool_msg)
    assert result["role"] == "assistant"
    assert "tool_calls" in result

    # Test system message conversion
    system_msg = SystemMessage(content="You are a helpful assistant.")
    result = _convert_message_to_dict(system_msg)
    assert result["role"] == "system"
    assert result["content"] == "You are a helpful assistant."

    # Test function message conversion
    function_msg = FunctionMessage(name="get_weather", content="Sunny")
    result = _convert_message_to_dict(function_msg)
    assert result["role"] == "function"
    assert result["name"] == "get_weather"
    assert result["content"] == "Sunny"

    # Test tool message conversion
    tool_msg = ToolMessage(tool_call_id="call_1", content="The weather is sunny")
    result = _convert_message_to_dict(tool_msg)
    assert result["role"] == "tool"
    assert result["tool_call_id"] == "call_1"
    assert result["content"] == "The weather is sunny"

    # Test chat message conversion
    chat_msg = ChatMessage(role="custom", content="Custom message")
    result = _convert_message_to_dict(chat_msg)
    assert result["role"] == "custom"
    assert result["content"] == "Custom message"

    # Test message with additional name parameter
    named_msg = AIMessage(content="Hello", additional_kwargs={"name": "helper"})
    result = _convert_message_to_dict(named_msg)
    assert result["role"] == "assistant"
    assert result["name"] == "helper"


def test_acompletion_with_retry() -> None:
    """Test acompletion_with_retry method with various scenarios."""
    import pytest
    from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
    from langchain_community.chat_models.litellm import acompletion_with_retry, ChatLiteLLM

    # Test successful async completion
    async def test_successful_completion():
        chat = ChatLiteLLM(model="test")  # type: ignore[call-arg]
        run_manager = AsyncCallbackManagerForLLMRun.get_noop_manager()
        response = await acompletion_with_retry(
            chat,
            run_manager=run_manager,
            messages=[{"role": "user", "content": "Hello"}]
        )
        assert isinstance(response, dict)
        assert "choices" in response

    # Test retry on error
    async def test_retry_on_error():
        chat = ChatLiteLLM(model="test", max_retries=2)  # type: ignore[call-arg]
        run_manager = AsyncCallbackManagerForLLMRun.get_noop_manager()
        
        # Mock client to simulate API error
        class MockClient:
            call_count = 0
            async def acreate(self, **kwargs):
                self.call_count += 1
                if self.call_count == 1:
                    raise Exception("API Error")
                return {"choices": [{"message": {"content": "Hello"}}]}
        
        chat.client = MockClient()
        response = await acompletion_with_retry(
            chat,
            run_manager=run_manager,
            messages=[{"role": "user", "content": "Hello"}]
        )
        assert isinstance(response, dict)
        assert "choices" in response
        assert chat.client.call_count == 2  # Verify retry happened

    # Run async tests
    import asyncio
    asyncio.run(test_successful_completion())
    asyncio.run(test_retry_on_error())


def test_lc_tool_call_to_openai_tool_call() -> None:
    """Test _lc_tool_call_to_openai_tool_call method with various scenarios."""
    from langchain_community.chat_models.litellm import _lc_tool_call_to_openai_tool_call

    # Test basic tool call conversion
    tool_call = {
        "id": "call_1",
        "name": "get_weather",
        "args": {"location": "London", "unit": "celsius"}
    }
    result = _lc_tool_call_to_openai_tool_call(tool_call)
    assert result["type"] == "function"
    assert result["id"] == "call_1"
    assert result["function"]["name"] == "get_weather"
    assert "location" in result["function"]["arguments"]
    assert "unit" in result["function"]["arguments"]

    # Test tool call with empty args
    tool_call_empty_args = {
        "id": "call_2",
        "name": "get_time",
        "args": {}
    }
    result = _lc_tool_call_to_openai_tool_call(tool_call_empty_args)
    assert result["type"] == "function"
    assert result["id"] == "call_2"
    assert result["function"]["name"] == "get_time"
    assert result["function"]["arguments"] == "{}"

    # Test tool call with nested args
    tool_call_nested = {
        "id": "call_3",
        "name": "search_database",
        "args": {
            "query": {
                "field": "name",
                "value": "John",
                "filters": ["active", "verified"]
            }
        }
    }
    result = _lc_tool_call_to_openai_tool_call(tool_call_nested)
    assert result["type"] == "function"
    assert result["id"] == "call_3"
    assert result["function"]["name"] == "search_database"
    assert "query" in result["function"]["arguments"]

    # Test different role conversions
    roles = [
        ("user", HumanMessageChunk),
        ("assistant", AIMessageChunk),
        ("system", SystemMessageChunk),
        ("function", FunctionMessageChunk),
        ("custom", ChatMessageChunk)
    ]

    for role, expected_class in roles:
        content = "Test content"
        if role == "function":
            chunk = _convert_delta_to_message_chunk(
                {"role": role, "content": content, "name": "test_func"},
                expected_class
            )
        else:
            chunk = _convert_delta_to_message_chunk(
                {"role": role, "content": content},
                expected_class
            )
        assert isinstance(chunk, expected_class)
        assert chunk.content == content

    # Test empty content
    empty_dict = {"role": "assistant"}
    chunk = _convert_delta_to_message_chunk(empty_dict, AIMessageChunk)
    assert isinstance(chunk, AIMessageChunk)
    assert chunk.content == ""

    # Test invalid tool_calls
    invalid_tool_calls_dict = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "call_1",
            "function": {}
        }]
    }
    chunk = _convert_delta_to_message_chunk(invalid_tool_calls_dict, AIMessageChunk)
    assert isinstance(chunk, AIMessageChunk)
    assert "tool_calls" in chunk.additional_kwargs


def test_convert_dict_to_message() -> None:
    """Test _convert_dict_to_message method with various scenarios."""
    from langchain_core.messages import (
        AIMessage,
        HumanMessage,
        SystemMessage,
        FunctionMessage,
        ChatMessage,
        ToolMessage
    )
    from langchain_community.chat_models.litellm import _convert_dict_to_message

    # Test user message
    user_dict = {"role": "user", "content": "Hello"}
    message = _convert_dict_to_message(user_dict)
    assert isinstance(message, HumanMessage)
    assert message.content == "Hello"

    # Test assistant message with function call
    assistant_dict = {
        "role": "assistant",
        "content": "Let me check the weather",
        "function_call": {"name": "get_weather", "arguments": "{\"location\": \"London\"}"}
    }
    message = _convert_dict_to_message(assistant_dict)
    assert isinstance(message, AIMessage)
    assert message.content == "Let me check the weather"
    assert message.additional_kwargs["function_call"]["name"] == "get_weather"

    # Test assistant message with tool calls
    assistant_tool_dict = {
        "role": "assistant",
        "content": "Let me help you",
        "tool_calls": [{
            "id": "call_1",
            "function": {"name": "get_weather", "arguments": "{\"location\": \"London\"}"}
        }]
    }
    message = _convert_dict_to_message(assistant_tool_dict)
    assert isinstance(message, AIMessage)
    assert "tool_calls" in message.additional_kwargs

    # Test system message
    system_dict = {"role": "system", "content": "You are a helpful assistant."}
    message = _convert_dict_to_message(system_dict)
    assert isinstance(message, SystemMessage)
    assert message.content == "You are a helpful assistant."

    # Test function message
    function_dict = {"role": "function", "name": "get_weather", "content": "Sunny"}
    message = _convert_dict_to_message(function_dict)
    assert isinstance(message, FunctionMessage)
    assert message.name == "get_weather"
    assert message.content == "Sunny"

    # Test tool message
    tool_dict = {"role": "tool", "tool_call_id": "call_1", "content": "The weather is sunny"}
    message = _convert_dict_to_message(tool_dict)
    assert isinstance(message, ToolMessage)
    assert message.tool_call_id == "call_1"
    assert message.content == "The weather is sunny"

    # Test custom role message
    custom_dict = {"role": "custom", "content": "Custom message"}
    message = _convert_dict_to_message(custom_dict)
    assert isinstance(message, ChatMessage)
    assert message.role == "custom"
    assert message.content == "Custom message"

    # Test message with additional name parameter
    named_dict = {"role": "assistant", "content": "Hello", "name": "helper"}
    message = _convert_dict_to_message(named_dict)
    assert isinstance(message, AIMessage)
    assert message.additional_kwargs["name"] == "helper"
