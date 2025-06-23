import os
import sys
import subprocess
import platform
import psutil
import logging
import shutil
from typing import Dict, Any, List, Optional, Union
import json
import re

class OSOperationsService:
    """Service for handling OS-level operations safely."""
    
    def __init__(self, allow_file_operations: bool = True, 
                 allow_process_execution: bool = False,
                 restricted_paths: List[str] = None):
        self.allow_file_operations = allow_file_operations
        self.allow_process_execution = allow_process_execution
        self.restricted_paths = restricted_paths or []
        self.logger = logging.getLogger("orchestrator.os_operations")
        
        # Add default system paths to restricted list for safety
        if not self.restricted_paths:
            self.restricted_paths = [
                "C:\\Windows",
                "C:\\Program Files",
                "C:\\Program Files (x86)",
                "/bin",
                "/sbin",
                "/usr/bin",
                "/usr/sbin",
                "/etc"
            ]
    
    def process_command(self, command: str) -> Dict[str, Any]:
        """
        Process a natural language command and route to the appropriate function.
        Enhanced with multilingual support.
        """
        command_lower = command.lower().strip()
        original_command = command.strip()
        
        self.logger.info(f"Processing OS command: {command_lower}")
        
        # File creation - ENHANCED MULTILINGUAL
        if any(pattern in command_lower for pattern in [
            # English patterns
            "create file", "create a file", "write file", "make file", 
            "new file", "save file", "create text file", "make text file",
            "write a file", "make a text file", "create a text file",
            "write hello", "hello world", "hello.txt",
            # French patterns - ADDED
            "créer fichier", "créer un fichier", "écrire fichier", 
            "nouveau fichier", "sauvegarder fichier", "créer fichier texte",
            "faire un fichier", "fichier avec", "avec", "contenant",
            "bonjour monde"
        ]):
            self.logger.info("Detected file creation command")
            path, content = self._extract_file_creation_info(original_command)
            return self.write_file(path, content)
        
        # Original patterns follow...
        elif any(x in command_lower for x in ["list files", "list directory", "show files"]):
            path = self._extract_path_from_command(command_lower)
            return self.list_directory(path)
        
        elif any(x in command_lower for x in ["read file", "show content", "display file"]):
            path = self._extract_path_from_command(command_lower)
            return self.read_file(path)
        
        elif any(x in command_lower for x in ["write file", "create file", "save to file"]):
            path = self._extract_path_from_command(command_lower)
            content = self._extract_content_from_command(command_lower)
            return self.write_file(path, content)
        
        elif any(x in command_lower for x in ["delete file", "remove file"]):
            path = self._extract_path_from_command(command_lower)
            return self.delete_file(path)
        
        elif any(x in command_lower for x in ["copy file", "duplicate file"]):
            source, destination = self._extract_source_dest_from_command(command_lower)
            return self.copy_file(source, destination)
        
        # System information
        elif any(x in command_lower for x in ["system info", "system information", "computer info"]):
            return self.get_system_info()
        
        elif any(x in command_lower for x in ["disk space", "storage info"]):
            return self.get_disk_space()
        
        elif any(x in command_lower for x in ["memory usage", "ram usage"]):
            return self.get_memory_usage()
        
        elif any(x in command_lower for x in ["cpu usage", "processor usage"]):
            return self.get_cpu_usage()
        
        # Process execution (if enabled)
        elif any(x in command_lower for x in ["run command", "execute command"]) and self.allow_process_execution:
            cmd = command_lower.split("run command", 1)[1].strip() if "run command" in command_lower else command_lower.split("execute command", 1)[1].strip()
            return self.execute_command(cmd)
        
        # Default response
        else:
            return {
                "success": False,
                "message": f"Command not recognized: {command}",
                "command_type": "unknown"
            }
    
    def _extract_path_from_command(self, command: str) -> str:
        """Extract file path from a natural language command."""
        # Look for path patterns
        if "path:" in command:
            path = command.split("path:", 1)[1].strip()
            path = path.split(" ", 1)[0] if " " in path else path
        elif "file:" in command:
            path = command.split("file:", 1)[1].strip()
            path = path.split(" ", 1)[0] if " " in path else path
        elif "directory:" in command:
            path = command.split("directory:", 1)[1].strip()
            path = path.split(" ", 1)[0] if " " in path else path
        else:
            # Default to current directory if no path specified
            path = "."
        
        # Handle quoted paths
        if (path.startswith('"') and path.endswith('"')) or (path.startswith("'") and path.endswith("'")):
            path = path[1:-1]
        
        return path
    
    def _extract_content_from_command(self, command: str) -> str:
        """Extract content to write from a natural language command."""
        if "content:" in command:
            return command.split("content:", 1)[1].strip()
        else:
            # Default content
            return "Content not specified in the command."
    
    def _extract_source_dest_from_command(self, command: str) -> tuple:
        """Extract source and destination paths from a command."""
        if "from:" in command and "to:" in command:
            source_part = command.split("from:", 1)[1]
            source = source_part.split("to:", 1)[0].strip()
            destination = command.split("to:", 1)[1].strip()
        else:
            # Default
            source = "source_not_specified"
            destination = "destination_not_specified"
        
        return source, destination
    
    def _is_path_allowed(self, path: str) -> bool:
        """Check if the given path is allowed for operations."""
        abs_path = os.path.abspath(path)
        
        # Check against restricted paths
        for restricted_path in self.restricted_paths:
            if abs_path.startswith(restricted_path):
                self.logger.warning(f"Attempted access to restricted path: {abs_path}")
                return False
        
        return True
    
    def list_directory(self, path: str = ".") -> Dict[str, Any]:
        """
        List contents of a directory.
        
        Args:
            path: Directory path to list
            
        Returns:
            Dict containing directory listing
        """
        if not self.allow_file_operations:
            return {"success": False, "message": "File operations not allowed", "command_type": "list_directory"}
        
        if not self._is_path_allowed(path):
            return {"success": False, "message": f"Access to path '{path}' is restricted", "command_type": "list_directory"}
        
        try:
            abs_path = os.path.abspath(path)
            if not os.path.exists(abs_path):
                return {"success": False, "message": f"Path does not exist: {abs_path}", "command_type": "list_directory"}
            
            if not os.path.isdir(abs_path):
                return {"success": False, "message": f"Path is not a directory: {abs_path}", "command_type": "list_directory"}
            
            # Get directory contents
            items = os.listdir(abs_path)
            
            # Categorize items
            directories = []
            files = []
            
            for item in items:
                item_path = os.path.join(abs_path, item)
                if os.path.isdir(item_path):
                    directories.append({
                        "name": item,
                        "type": "directory"
                    })
                else:
                    # Get file size
                    size = os.path.getsize(item_path)
                    files.append({
                        "name": item,
                        "type": "file",
                        "size": size,
                        "size_human": self._format_size(size)
                    })
            
            return {
                "success": True,
                "message": f"Listed directory: {abs_path}",
                "command_type": "list_directory",
                "path": abs_path,
                "contents": {
                    "directories": directories,
                    "files": files,
                    "total_items": len(directories) + len(files)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error listing directory {path}: {e}")
            return {"success": False, "message": f"Error listing directory: {str(e)}", "command_type": "list_directory"}
    
    def read_file(self, path: str) -> Dict[str, Any]:
        """
        Read contents of a file.
        
        Args:
            path: File path to read
            
        Returns:
            Dict containing file contents
        """
        if not self.allow_file_operations:
            return {"success": False, "message": "File operations not allowed", "command_type": "read_file"}
        
        if not self._is_path_allowed(path):
            return {"success": False, "message": f"Access to path '{path}' is restricted", "command_type": "read_file"}
        
        try:
            abs_path = os.path.abspath(path)
            if not os.path.exists(abs_path):
                return {"success": False, "message": f"File does not exist: {abs_path}", "command_type": "read_file"}
            
            if not os.path.isfile(abs_path):
                return {"success": False, "message": f"Path is not a file: {abs_path}", "command_type": "read_file"}
            
            # Check file size - don't read huge files
            file_size = os.path.getsize(abs_path)
            if file_size > 10 * 1024 * 1024:  # 10 MB limit
                return {
                    "success": False, 
                    "message": f"File is too large to read ({self._format_size(file_size)}). Limit is 10 MB.", 
                    "command_type": "read_file"
                }
            
            # Try to detect if it's a binary file
            is_binary = False
            try:
                with open(abs_path, 'r', encoding='utf-8') as test_f:
                    test_f.read(1024)  # Try reading as text
            except UnicodeDecodeError:
                is_binary = True
            
            if is_binary:
                return {
                    "success": False,
                    "message": f"File appears to be binary and cannot be displayed as text: {abs_path}",
                    "command_type": "read_file"
                }
            
            # Read the file
            with open(abs_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            return {
                "success": True,
                "message": f"Read file: {abs_path}",
                "command_type": "read_file",
                "path": abs_path,
                "content": content,
                "size": file_size,
                "size_human": self._format_size(file_size)
            }
            
        except Exception as e:
            self.logger.error(f"Error reading file {path}: {e}")
            return {"success": False, "message": f"Error reading file: {str(e)}", "command_type": "read_file"}
    
    def write_file(self, path: str, content: str) -> Dict[str, Any]:
        """
        Write content to a file.
        
        Args:
            path: File path to write
            content: Content to write
            
        Returns:
            Dict containing operation result
        """
        if not self.allow_file_operations:
            return {"success": False, "message": "File operations not allowed", "command_type": "write_file"}
        
        if not self._is_path_allowed(path):
            return {"success": False, "message": f"Access to path '{path}' is restricted", "command_type": "write_file"}
        
        try:
            abs_path = os.path.abspath(path)
            directory = os.path.dirname(abs_path)
            
            # Ensure directory exists
            if not os.path.exists(directory):
                os.makedirs(directory)
            
            # Write the file
            with open(abs_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            file_size = os.path.getsize(abs_path)
            
            return {
                "success": True,
                "message": f"Wrote file: {abs_path}",
                "command_type": "write_file",
                "path": abs_path,
                "size": file_size,
                "size_human": self._format_size(file_size)
            }
            
        except Exception as e:
            self.logger.error(f"Error writing file {path}: {e}")
            return {"success": False, "message": f"Error writing file: {str(e)}", "command_type": "write_file"}
    
    def delete_file(self, path: str) -> Dict[str, Any]:
        """
        Delete a file.
        
        Args:
            path: File path to delete
            
        Returns:
            Dict containing operation result
        """
        if not self.allow_file_operations:
            return {"success": False, "message": "File operations not allowed", "command_type": "delete_file"}
        
        if not self._is_path_allowed(path):
            return {"success": False, "message": f"Access to path '{path}' is restricted", "command_type": "delete_file"}
        
        try:
            abs_path = os.path.abspath(path)
            if not os.path.exists(abs_path):
                return {"success": False, "message": f"File does not exist: {abs_path}", "command_type": "delete_file"}
            
            # Check if it's a file or directory
            if os.path.isfile(abs_path):
                os.remove(abs_path)
                operation_type = "file"
            elif os.path.isdir(abs_path):
                shutil.rmtree(abs_path)
                operation_type = "directory"
            else:
                return {"success": False, "message": f"Path is not a file or directory: {abs_path}", "command_type": "delete_file"}
            
            return {
                "success": True,
                "message": f"Deleted {operation_type}: {abs_path}",
                "command_type": "delete_file",
                "path": abs_path,
                "type": operation_type
            }
            
        except Exception as e:
            self.logger.error(f"Error deleting {path}: {e}")
            return {"success": False, "message": f"Error deleting file: {str(e)}", "command_type": "delete_file"}
    
    def copy_file(self, source: str, destination: str) -> Dict[str, Any]:
        """
        Copy a file from source to destination.
        
        Args:
            source: Source file path
            destination: Destination file path
            
        Returns:
            Dict containing operation result
        """
        if not self.allow_file_operations:
            return {"success": False, "message": "File operations not allowed", "command_type": "copy_file"}
        
        if not self._is_path_allowed(source) or not self._is_path_allowed(destination):
            return {"success": False, "message": "Access to path is restricted", "command_type": "copy_file"}
        
        try:
            abs_source = os.path.abspath(source)
            abs_dest = os.path.abspath(destination)
            
            if not os.path.exists(abs_source):
                return {"success": False, "message": f"Source does not exist: {abs_source}", "command_type": "copy_file"}
            
            # Ensure destination directory exists
            dest_dir = os.path.dirname(abs_dest)
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            
            # Check if source is file or directory
            if os.path.isfile(abs_source):
                shutil.copy2(abs_source, abs_dest)
                operation_type = "file"
            elif os.path.isdir(abs_source):
                if os.path.exists(abs_dest):
                    shutil.rmtree(abs_dest)
                shutil.copytree(abs_source, abs_dest)
                operation_type = "directory"
            else:
                return {"success": False, "message": f"Source is not a file or directory: {abs_source}", "command_type": "copy_file"}
            
            return {
                "success": True,
                "message": f"Copied {operation_type} from {abs_source} to {abs_dest}",
                "command_type": "copy_file",
                "source": abs_source,
                "destination": abs_dest,
                "type": operation_type
            }
            
        except Exception as e:
            self.logger.error(f"Error copying from {source} to {destination}: {e}")
            return {"success": False, "message": f"Error copying file: {str(e)}", "command_type": "copy_file"}
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get system information.
        
        Returns:
            Dict containing system information
        """
        try:
            system_info = {
                "system": platform.system(),
                "node": platform.node(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python_version": sys.version,
                "cpu_count": psutil.cpu_count(logical=False),
                "logical_cpu_count": psutil.cpu_count(logical=True),
                "memory_total": psutil.virtual_memory().total,
                "memory_total_human": self._format_size(psutil.virtual_memory().total)
            }
            
            return {
                "success": True,
                "message": "Retrieved system information",
                "command_type": "system_info",
                "system_info": system_info
            }
            
        except Exception as e:
            self.logger.error(f"Error getting system information: {e}")
            return {"success": False, "message": f"Error getting system information: {str(e)}", "command_type": "system_info"}
    
    def get_disk_space(self) -> Dict[str, Any]:
        """
        Get disk space information.
        
        Returns:
            Dict containing disk space information
        """
        try:
            disk_info = []
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    disk_info.append({
                        "device": partition.device,
                        "mountpoint": partition.mountpoint,
                        "filesystem": partition.fstype,
                        "total": usage.total,
                        "total_human": self._format_size(usage.total),
                        "used": usage.used,
                        "used_human": self._format_size(usage.used),
                        "free": usage.free,
                        "free_human": self._format_size(usage.free),
                        "percent_used": usage.percent
                    })
                except (PermissionError, OSError):
                    # Skip partitions that can't be accessed
                    pass
            
            return {
                "success": True,
                "message": "Retrieved disk space information",
                "command_type": "disk_space",
                "disk_info": disk_info
            }
            
        except Exception as e:
            self.logger.error(f"Error getting disk space: {e}")
            return {"success": False, "message": f"Error getting disk space: {str(e)}", "command_type": "disk_space"}
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get memory usage information.
        
        Returns:
            Dict containing memory usage information
        """
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            memory_info = {
                "total": memory.total,
                "total_human": self._format_size(memory.total),
                "available": memory.available,
                "available_human": self._format_size(memory.available),
                "used": memory.used,
                "used_human": self._format_size(memory.used),
                "percent_used": memory.percent,
                "swap_total": swap.total,
                "swap_total_human": self._format_size(swap.total),
                "swap_used": swap.used,
                "swap_used_human": self._format_size(swap.used),
                "swap_percent_used": swap.percent
            }
            
            return {
                "success": True,
                "message": "Retrieved memory usage information",
                "command_type": "memory_usage",
                "memory_info": memory_info
            }
            
        except Exception as e:
            self.logger.error(f"Error getting memory usage: {e}")
            return {"success": False, "message": f"Error getting memory usage: {str(e)}", "command_type": "memory_usage"}
    
    def get_cpu_usage(self) -> Dict[str, Any]:
        """
        Get CPU usage information.
        
        Returns:
            Dict containing CPU usage information
        """
        try:
            # Get CPU usage percentage (across all cores)
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Get per-core CPU usage
            per_cpu = psutil.cpu_percent(interval=0.1, percpu=True)
            
            # Get CPU frequency
            freq = psutil.cpu_freq()
            cpu_freq = {
                "current": None,
                "min": None,
                "max": None
            }
            
            if freq:
                cpu_freq["current"] = freq.current
                cpu_freq["min"] = freq.min
                cpu_freq["max"] = freq.max
            
            # Get CPU stats
            cpu_stats = psutil.cpu_stats()
            
            # Get CPU load averages (Linux/macOS)
            load_avg = None
            if hasattr(psutil, "getloadavg"):
                load_avg = psutil.getloadavg()
            
            cpu_info = {
                "percent": cpu_percent,
                "per_cpu_percent": per_cpu,
                "frequency": cpu_freq,
                "stats": {
                    "ctx_switches": cpu_stats.ctx_switches,
                    "interrupts": cpu_stats.interrupts,
                    "soft_interrupts": cpu_stats.soft_interrupts,
                    "syscalls": cpu_stats.syscalls
                }
            }
            
            if load_avg:
                cpu_info["load_avg"] = {
                    "1min": load_avg[0],
                    "5min": load_avg[1],
                    "15min": load_avg[2]
                }
            
            return {
                "success": True,
                "message": "Retrieved CPU usage information",
                "command_type": "cpu_usage",
                "cpu_info": cpu_info
            }
            
        except Exception as e:
            self.logger.error(f"Error getting CPU usage: {e}")
            return {"success": False, "message": f"Error getting CPU usage: {str(e)}", "command_type": "cpu_usage"}
    
    def execute_command(self, command: str) -> Dict[str, Any]:
        """
        Execute a system command (if allowed).
        
        Args:
            command: Command to execute
            
        Returns:
            Dict containing command output
        """
        if not self.allow_process_execution:
            return {"success": False, "message": "Process execution not allowed", "command_type": "execute_command"}
        
        # Block potentially dangerous commands
        dangerous_commands = ["rm", "del", "format", "mkfs", "dd", "shutdown", "reboot", ">", "sudo", "su"]
        if any(cmd in command.lower() for cmd in dangerous_commands):
            return {
                "success": False, 
                "message": f"Potentially dangerous command not allowed: {command}", 
                "command_type": "execute_command"
            }
        
        try:
            # Execute command and capture output
            process = subprocess.Popen(
                command, 
                shell=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Set a timeout to prevent hanging
            try:
                stdout, stderr = process.communicate(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                return {
                    "success": False,
                    "message": "Command execution timed out after 30 seconds",
                    "command_type": "execute_command",
                    "command": command
                }
            
            return {
                "success": process.returncode == 0,
                "message": f"Command executed with return code {process.returncode}",
                "command_type": "execute_command",
                "command": command,
                "return_code": process.returncode,
                "stdout": stdout,
                "stderr": stderr
            }
            
        except Exception as e:
            self.logger.error(f"Error executing command {command}: {e}")
            return {"success": False, "message": f"Error executing command: {str(e)}", "command_type": "execute_command"}
    
    def _format_size(self, size_bytes: int) -> str:
        """Format bytes into a human-readable string."""
        if size_bytes == 0:
            return "0B"
        
        size_names = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024
            i += 1
        
        return f"{size_bytes:.2f} {size_names[i]}"
    
    def _extract_file_creation_info(self, command: str) -> tuple:
        """
        Extract filename and content from natural language file creation commands.
        Improved with multilingual support (English and French).
        """
        command_lower = command.lower().strip()
        
        # Default values as last resort
        filename = None
        content = None
        
        self.logger.info(f"Extracting file info from: '{command}'")
        
        # CONTENT EXTRACTION - ENHANCED MULTILINGUAL
        
        # Check for quoted content (highest priority)
        quote_matches = re.findall(r'["\']([^"\']+)["\']', command)
        if quote_matches:
            content = quote_matches[0]
            self.logger.info(f"Found quoted content: '{content}'")
        
        # English patterns
        elif "with" in command_lower and ("in it" in command_lower or "containing" in command_lower):
            start_idx = command_lower.find("with") + 4
            end_idx = command_lower.find("in it") if "in it" in command_lower else command_lower.find("containing")
            if start_idx > 4 and end_idx > start_idx:
                content = command[start_idx:end_idx].strip()
        
        # French patterns - ADDED
        elif "avec" in command_lower:
            start_idx = command_lower.find("avec") + 4
            end_idx = len(command_lower)
            # Look for end markers in French
            for marker in ["dedans", "dans", "à l'intérieur"]:
                if marker in command_lower[start_idx:]:
                    end_idx = command_lower.find(marker, start_idx)
                    break
            content = command[start_idx:end_idx].strip()
        
        # Content after specific markers
        elif "content:" in command_lower:
            content = command.split("content:", 1)[1].strip()
        elif "contenu:" in command_lower:  # French
            content = command.split("contenu:", 1)[1].strip()
        elif ":" in command and not any(marker in command_lower for marker in ["file:", "path:", "name:", "fichier:", "nom:"]):
            parts = command.split(":", 1)
            if len(parts) > 1:
                content = parts[1].strip()
        
        # Look for content after "containing" or "contenant" (French)
        elif "containing" in command_lower:
            content = command.split("containing", 1)[1].strip()
        elif "contenant" in command_lower:
            content = command.split("contenant", 1)[1].strip()
        
        # FILENAME EXTRACTION - ENHANCED MULTILINGUAL
        
        # Check for explicit filename specifiers
        if "file:" in command_lower:
            filename_part = command.split("file:", 1)[1].strip()
            filename = filename_part.split()[0]
        elif "fichier:" in command_lower:  # French
            filename_part = command.split("fichier:", 1)[1].strip()
            filename = filename_part.split()[0]
        elif "name:" in command_lower:
            filename_part = command.split("name:", 1)[1].strip()
            filename = filename_part.split()[0]
        elif "nom:" in command_lower:  # French
            filename_part = command.split("nom:", 1)[1].strip()
            filename = filename_part.split()[0]
        
        # Common phrases indicating filename
        elif "called" in command_lower:
            called_idx = command_lower.find("called") + 6
            filename_part = command[called_idx:].strip().split()[0]
            if filename_part:
                filename = filename_part
        elif "named" in command_lower:
            named_idx = command_lower.find("named") + 5
            filename_part = command[named_idx:].strip().split()[0]
            if filename_part:
                filename = filename_part
        # French variants - ADDED
        elif "appelé" in command_lower or "nommé" in command_lower:
            idx = command_lower.find("appelé") if "appelé" in command_lower else command_lower.find("nommé")
            idx += 6  # Length of "appelé" or "nommé"
            filename_part = command[idx:].strip().split()[0]
            if filename_part:
                filename = filename_part
                
        if content is None:
            content = "Hello World"
        
        if filename is None:
            filename = "hello.txt"
        
        # Clean up extracted values
        filename = filename.strip('"\'.,;: ')
        content = content.strip('"\'.,;: ')
        
        # Ensure filename has an extension
        if "." not in filename:
            filename = filename + ".txt"
        
        self.logger.info(f"Final extracted: filename='{filename}', content='{content[:20]}...'")
        return filename, content