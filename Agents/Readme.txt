#Agents

Agents can be used for a variety of tasks. Agents combine the decision making ability of a language model with tools
in order to create a system that can execute and implement solutions on your behalf. Before reading any more, it is
highly recommended that you read the documentation in the agent module to understand the concepts associated with agents more. 
Specifically, you should be familiar with what the agent, tool, and agent executor abstractions are before reading more.

Agent Documentation (for interacting with the outside world)

#Create Your Own Agent
Once you have read that documentation, you should be prepared to create your own agent.
What exactly does that involve? Here’s how we recommend getting started with creating your own agent:

#Step 1: Create Tools
Agents are largely defined by the tools they can use. If you have a specific task you want the agent to accomplish, 
you have to give it access to the right tools. We have many tools natively in LangChain, so you should first look to 
see if any of them meet your needs. But we also make it easy to define a custom tool, so if you need custom tools you 
should absolutely do that.

#(Optional) Step 2: Modify Agent
The built-in LangChain agent types are designed to work well in generic situations, but you may be able to improve performance 
by modifying the agent implementation. There are several ways you could do this:

1.Modify the base prompt. This can be used to give the agent more context on how it should behave, etc.

2.Modify the output parser. This is necessary if the agent is having trouble parsing the language model output.

#Examples
Specific examples of agents include:

1.AI Plugins: an implementation of an agent that is designed to be able to use all AI Plugins.

2.Plug-and-PlAI (Plugins Database): an implementation of an agent that is designed to be able to use all AI Plugins retrieved from PlugNPlAI.

3.Wikibase Agent: an implementation of an agent that is designed to interact with Wikibase.

4.Sales GPT: This notebook demonstrates an implementation of a Context-Aware AI Sales agent.

5.Multi-Modal Output Agent: an implementation of a multi-modal output agent that can generate text and images.